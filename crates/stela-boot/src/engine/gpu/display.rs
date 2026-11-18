//! The guest's display on a bound GPU's output.
//!
//! Binding a GPU ends the firmware's framebuffer on it, so Stela's console
//! moves to the output's front buffer at once, where the modeled DCP's frames
//! follow it. A GPU driving no display leaves the firmware's scanout alone.
//!
//! The DCP model presents under the board's device lock, on whichever PE
//! made the device access. A DRM request from a PE other than the boot CPU
//! waits for the boot CPU to run Linux, and the boot CPU may be waiting for
//! that lock, so a present only copies rows into a hidden buffer there and
//! queues its page flip. The flip is requested on the boot CPU: by its own
//! presents, or by the display's flush at the next vblank, which the home
//! machine's device deadline brings there.
//!
//! Hot-plug: once a second the same flush on the boot CPU reads the
//! connectors as the driver last detected them, which its hot-plug
//! interrupts keep current. A display taken at start is set again when it
//! returns after being unplugged, and shows on an output connected later
//! when that output has its size or a larger native mode to scale it into.
//! When the firmware's screen is gone and no
//! display was taken, the bound GPUs are looked on until one shows a display
//! in the guest's mode, and probed anew every [`PROBE_LOOKS`] looks, as
//! Linux's output polling and fbdev helper take a display that appears late.
use super::client;
use crate::console::{self, Framebuffer};
use crate::engine::host;
use crate::{Result, output};
use alloc::{boxed::Box, format, string::String, vec::Vec};
use core::ops::Range;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use core::{cell::UnsafeCell, ffi::c_int, ffi::c_void, ptr::NonNull};
use stela_boot_core::options::DisplaySize;
use stela_contracts::refuse;
use stela_drivers::gpu::metal::kms::{self, Display, Drm, Error as DisplayError, Watched};
use stela_soc::machine::{MonitorId, Scanout, ScanoutFormat, ScanoutMode};

/// A GPU's primary node, the DRM master.
struct Node(c_int);

// SAFETY: Linux's DRM ioctls honor their UAPI bounds. `client::map` retains
// each mapping for the boot, and different dumb objects have distinct backing.
unsafe impl Drm for Node {
    fn control(&mut self, request: u32, argument: *mut c_void) -> i64 {
        client::control(self.0, request, argument)
    }

    fn map(&mut self, offset: u64, bytes: usize) -> Option<NonNull<u8>> {
        client::map(self.0, offset, bytes)
    }

    fn unmap(&mut self, pixels: NonNull<u8>, _: usize) {
        client::unmap(pixels);
    }

    fn immediate(&self) -> bool {
        on_boot_cpu()
    }
}

fn on_boot_cpu() -> bool {
    host::processor() == super::BOOT_CPU.load(Ordering::Relaxed)
}

/// How often the scanout reads its GPUs' connectors as the driver last
/// detected them ([`kms::detected`]).
const LOOK_SECONDS: u64 = 1;
/// Every this many looks, GPUs with no display taken have their connectors
/// probed anew, as Linux's output polling does every 10 s for connectors
/// whose hot-plug it is not told of.
const PROBE_LOOKS: u32 = 10;

/// The host scanout the modeled DCP presents to: a bound GPU's display, or,
/// when none was taken, the bound GPUs' open nodes, looked on until one
/// shows a display in the guest's mode. Looks run on the boot CPU only,
/// where a DRM request completes at once.
struct KmsScanout {
    /// The guest's mode, fixed when the DCP was placed.
    mode: ScanoutMode,
    refresh_millihertz: u32,
    size_mm: Option<(u32, u32)>,
    /// The identity of the monitor the guest's display was made for: the
    /// taken output's, or the firmware screen's while one is awaited.
    monitor: Option<MonitorId>,
    /// The GPU showing the display, and the display.
    display: Option<(String, Display<Node>)>,
    /// Without a display: the GPUs' open nodes to look on.
    waiting: Vec<(String, c_int)>,
    /// A frame was presented: a display taken later shows the guest's
    /// frames rather than Stela's console.
    presented: bool,
    /// A display was taken after a frame was presented and has shown none
    /// yet: the DCP model presents the frame on screen again, whole
    /// ([`Scanout::wants_frame`]), as macOS draws nothing new on a still
    /// screen until input.
    blank: bool,
    /// The counter at which the next look is due, and the looks so far.
    next_look: u64,
    looks: u32,
    tsc_hz: u64,
    /// A refused look was reported.
    reported: bool,
    /// The counter at the display's last vertical blank as the last look
    /// read it, when it could ([`Display::last_vblank_ns`]).
    vblank_tsc: Option<u64>,
}

impl Scanout for KmsScanout {
    fn mode(&self) -> ScanoutMode {
        self.mode
    }

    fn refresh_millihertz(&self) -> u32 {
        self.refresh_millihertz
    }

    fn physical_size_mm(&self) -> Option<(u32, u32)> {
        self.size_mm
    }

    fn monitor(&self) -> Option<MonitorId> {
        self.monitor
    }

    fn present(&mut self, rows: &[u8]) -> Result<()> {
        self.presented = true;
        let Some((_, display)) = &mut self.display else {
            // Shown nowhere: a display taken later asks every row of the
            // next frame, its buffers holding none.
            self.look();
            return Ok(());
        };
        console::release();
        let presented = display.present(rows);
        self.blank &= presented.is_err();
        finish_present(display, presented)
    }

    /// Without a display a frame is taken whole and shown nowhere, its rows
    /// unasked, as in [`Self::present`].
    fn try_present_rows(
        &mut self,
        damage: Option<Range<usize>>,
        fill: &mut dyn FnMut(usize, &mut [u8]) -> Result<()>,
    ) -> Result<bool> {
        self.presented = true;
        let Some((_, display)) = &mut self.display else {
            self.look();
            return Ok(true);
        };
        let prepared = display.prepare_rows(damage, fill)?;
        console::release();
        let presented = prepared.present();
        self.blank &= presented.is_err();
        finish_present(display, presented)?;
        Ok(true)
    }

    fn refresh_rates(&self) -> Vec<u32> {
        match &self.display {
            Some((_, display)) => display.refresh_rates(),
            None => alloc::vec![self.refresh_millihertz],
        }
    }

    /// Where a request must wait, the next flush on the boot CPU sets the
    /// mode; the vblanks take the new period at once.
    fn set_refresh(&mut self, millihertz: u32) -> Result<()> {
        let Some((_, display)) = &mut self.display else {
            return Err(refuse!("no GPU display runs the guest's mode"));
        };
        display.set_refresh(millihertz).map_err(|error| {
            refuse!(
                "the GPU's display refused {millihertz} mHz: {error}",
                error = format!("{error:?}")
            )
        })?;
        self.refresh_millihertz = millihertz;
        // The blank read before runs on the old period.
        self.vblank_tsc = None;
        Ok(())
    }

    /// The display's vertical blanks run on from the one the last look
    /// read; the model takes the age past whole periods.
    fn vblank_age_ns(&self) -> Option<u64> {
        let tsc = self.vblank_tsc?;
        let elapsed = host::rdtsc().saturating_sub(tsc);
        u64::try_from(u128::from(elapsed) * 1_000_000_000 / u128::from(self.tsc_hz.max(1))).ok()
    }

    fn queued(&self) -> bool {
        let looking = self.display.is_some() || !self.waiting.is_empty();
        (looking && host::rdtsc() >= self.next_look)
            || self
                .display
                .as_ref()
                .is_some_and(|(_, display)| display.queued())
    }

    fn flush(&mut self) -> Result<()> {
        if let Some((_, display)) = &mut self.display {
            let flushed = display.flush();
            finish_present(display, flushed)?;
        }
        self.look();
        Ok(())
    }

    /// Display sleep turns the outputs off, and a wake lights them again;
    /// where a request must wait, the next flush on the boot CPU does it.
    /// With no display taken nothing of this scanout is lit.
    fn wants_frame(&self) -> bool {
        self.blank
    }

    fn set_power(&mut self, on: bool) -> Result<()> {
        let Some((_, display)) = &mut self.display else {
            return Ok(());
        };
        display.set_power(on).map_err(|error| {
            refuse!(
                "the GPU's display refused to turn its outputs {state}: {error}",
                state = if on { "on" } else { "off" },
                error = format!("{error:?}")
            )
        })
    }
}

impl KmsScanout {
    fn new(
        mode: ScanoutMode,
        refresh_millihertz: u32,
        size_mm: Option<(u32, u32)>,
        monitor: Option<MonitorId>,
        tsc_hz: u64,
    ) -> Self {
        Self {
            mode,
            refresh_millihertz,
            size_mm,
            monitor,
            display: None,
            waiting: Vec::new(),
            presented: false,
            blank: false,
            next_look: host::rdtsc().saturating_add(tsc_hz.saturating_mul(LOOK_SECONDS)),
            looks: 0,
            tsc_hz,
            reported: false,
            vblank_tsc: None,
        }
    }

    /// When a look is due and this is the boot CPU: watch the display's
    /// connector, else look for a display on the waiting GPUs.
    fn look(&mut self) {
        let now = host::rdtsc();
        if now < self.next_look || !on_boot_cpu() {
            return;
        }
        self.next_look = now.saturating_add(self.tsc_hz.saturating_mul(LOOK_SECONDS));
        self.looks = self.looks.wrapping_add(1);
        match &mut self.display {
            Some((name, display)) => {
                // Linux's clock and the counter read together date the
                // blank on the counter.
                self.vblank_tsc = display.last_vblank_ns().and_then(|blank| {
                    let (tsc, now) = (host::rdtsc(), client::now_ns());
                    let age = u128::from(now.checked_sub(blank)?);
                    let ticks = age * u128::from(self.tsc_hz) / 1_000_000_000;
                    Some(tsc.saturating_sub(u64::try_from(ticks).ok()?))
                });
                let watched = display.watch();
                report_watched(name, display, watched);
                if let Some(error) = watched.error
                    && !core::mem::replace(&mut self.reported, true)
                {
                    output::arguments(format_args!(
                        "GPU {name}: display request refused: {error:?}; the other outputs \
                         are still read and set\n"
                    ));
                }
            }
            None => self.attach(self.looks.is_multiple_of(PROBE_LOOKS)),
        }
    }

    /// Take the first waiting GPU's display in the guest's mode, reading
    /// each GPU's connectors as last detected unless `probe`.
    fn attach(&mut self, probe: bool) {
        let wanted = (self.mode.width, self.mode.height);
        let mut index = 0;
        while let Some((name, fd)) = self.waiting.get(index) {
            let fd = *fd;
            if !probe && kms::detected(&mut Node(fd)) != Ok(true) {
                index += 1;
                continue;
            }
            match Display::take_fitting(
                Node(fd),
                kms::Wanted::Size(wanted),
                &stela_soc::dcp::modes::advertisable,
            ) {
                Ok(display) if display.size() == wanted => {
                    output::arguments(format_args!("GPU {name}: display detected after start\n"));
                    report_ports(name, &display);
                    report_outputs(name, &display);
                    let screen = screen(&display);
                    match self.presented {
                        // The guest's frame on screen shows here at once,
                        // presented again, and a stop's reason.
                        true => console::parked_on(screen),
                        false => crate::status::take_screen(Some(screen)),
                    }
                    self.blank = self.presented;
                    let (name, _) = self.waiting.swap_remove(index);
                    for (_, fd) in self.waiting.drain(..) {
                        client::close_descriptor(fd);
                    }
                    self.display = Some((name, display));
                    self.reported = false;
                    return;
                }
                Ok(display) => {
                    // No output runs the guest's size itself, which the
                    // output that gives the picture's timing must; the
                    // outputs are turned off, their buffers and the node
                    // given back, and the GPU is looked on no more.
                    let (width, height) = display.size();
                    output::arguments(format_args!(
                        "GPU {name}: display detected after start has no {}x{} mode, \
                         which macOS draws, only {width}x{height}: not shown\n",
                        wanted.0, wanted.1
                    ));
                    let Node(fd) = display.release();
                    client::close_descriptor(fd);
                    self.waiting.swap_remove(index);
                }
                Err(DisplayError::NoOutput) => index += 1,
                Err(error) => {
                    if !core::mem::replace(&mut self.reported, true) {
                        output::arguments(format_args!(
                            "GPU {name}: display look refused: {error:?}\n"
                        ));
                    }
                    index += 1;
                }
            }
        }
    }
}

/// What a look read of `display`'s connectors, one line for each change.
fn report_watched(name: &str, display: &Display<Node>, watched: Watched) {
    let lines = [
        (watched.lost, "display disconnected"),
        (watched.restored, "display connected again"),
        (watched.joined, "display added on a new output"),
        (
            watched.scaled,
            "added or returned output has no mode of the display's size: picture scaled into \
             its native mode",
        ),
        (
            watched.unshown,
            "connected output still has no free mode or controller in the display's size: \
             left dark",
        ),
        (
            watched.modeless,
            "display connected again has no mode of the display's size: left dark",
        ),
        (
            watched.smaller,
            "dark output's native mode is smaller than the display's size, and the picture is \
             only scaled up",
        ),
        (
            watched.refused,
            "display connected again refused its mode: left dark",
        ),
    ];
    for (count, what) in lines {
        if count != 0 {
            output::arguments(format_args!(
                "GPU {name}: {what} ({count}); {} outputs run it\n",
                display.outputs()
            ));
        }
    }
}

/// Whether a display taken shows on a laptop's built-in panel, whose
/// backlight is set apart from the picture ([`panel_shown`]).
static PANEL: AtomicBool = AtomicBool::new(false);

/// Whether a GPU's display shows on a built-in panel (eDP, LVDS or DSI).
pub(in crate::engine) fn panel_shown() -> bool {
    PANEL.load(Ordering::Relaxed)
}

/// The connectors `display` shows on, the one that paces it first, which
/// tells a laptop panel on this GPU (eDP) from an external monitor.
fn report_ports(name: &str, display: &Display<Node>) {
    let mut ports = String::new();
    for port in display.ports() {
        if !ports.is_empty() {
            ports.push_str(", ");
        }
        let _ = core::fmt::write(&mut ports, format_args!("{port}"));
        if port.panel() {
            PANEL.store(true, Ordering::Relaxed);
            ports.push_str(" (built-in panel)");
        }
    }
    output::arguments(format_args!("GPU {name}: display shows on {ports}\n"));
}

/// How many outputs show `display` and how many connected ones cannot,
/// when that is more than the one output, then each output that scales the
/// picture and why others stay dark.
fn report_outputs(name: &str, display: &Display<Node>) {
    let (shown, unshown) = (display.outputs(), display.unshown());
    if shown <= 1 && unshown == 0 {
        return;
    }
    let (width, height) = display.size();
    output::arguments(format_args!(
        "GPU {name}: display on {shown} outputs, {unshown} connected without a free \
         {width}x{height} mode or controller\n"
    ));
    for kms::Scaled { native, picture } in display.scaled() {
        output::arguments(format_args!(
            "GPU {name}: an output has no {width}x{height} mode: it runs {}x{} and the picture \
             fills {}x{} of it\n",
            native.0, native.1, picture.0, picture.1
        ));
    }
    let smaller = display.smaller();
    if smaller != 0 {
        output::arguments(format_args!(
            "GPU {name}: {smaller} dark outputs have a native mode smaller than \
             {width}x{height}; the picture is only scaled up\n"
        ));
    }
    if let Some(error) = display.refused() {
        output::arguments(format_args!(
            "GPU {name}: scaling an output was refused: {error:?}; it stays dark\n"
        ));
    }
}

fn finish_present(display: &Display<Node>, presented: Result<(), DisplayError>) -> Result<()> {
    // A stop's reason shows on the buffer that does.
    console::parked_on(screen(display));
    presented.map_err(|error| {
        refuse!(
            "the GPU's display refused a frame: {error}",
            error = format!("{error:?}")
        )
    })
}

/// `display`'s front buffer as a screen.
fn screen(display: &Display<Node>) -> Framebuffer {
    let (width, height) = display.size();
    let (pixels, pitch) = display.front();
    Framebuffer::linear(
        pixels.as_ptr() as usize,
        width as usize,
        height as usize,
        pitch / size_of::<u32>(),
    )
}

/// What [`take`] left, until the board's DCP takes it: the display taken,
/// else the nodes of the GPUs that had none.
struct Taken(UnsafeCell<Displays>);

/// The display a GPU took with that GPU's name, and the named nodes of the
/// GPUs still without one.
type Displays = (Option<(String, Display<Node>)>, Vec<(String, c_int)>);

// SAFETY: only the boot CPU's start reaches it, once each way.
unsafe impl Sync for Taken {}

static TAKEN: Taken = Taken(UnsafeCell::new((None, Vec::new())));
/// The bound GPU whose output [`take`] took, or `usize::MAX`.
static TAKEN_INDEX: AtomicUsize = AtomicUsize::new(usize::MAX);

/// The bound GPU, by its index, whose output took the connected display.
pub(super) fn taken_index() -> Option<usize> {
    let index = TAKEN_INDEX.load(Ordering::Relaxed);
    (index != usize::MAX).then_some(index)
}

/// Which GPU showed the firmware's picture.
#[derive(Clone, Copy)]
pub(in crate::engine) enum Shown<'a> {
    /// The firmware left no screen.
    Nothing,
    /// A screen whose PCI function was not found.
    Unknown,
    /// This function's.
    By(&'a str),
}

/// How long the GPU that showed the firmware's picture may take to detect
/// its display again. The GPU's reset ends the signal, and a monitor that
/// then searches its inputs drops hot-plug detect for some seconds; Linux
/// takes such a display when its hot-plug event arrives. A display later
/// still is taken by the scanout's looks.
const RETURN_SECONDS: u64 = 8;
/// The pause between two detections in that time.
const RETURN_STEP_MILLIS: u64 = 100;

/// Moves the display to the first bound GPU with an output, the GPUs that
/// draw first and then those that only show (`claim_all`'s order), in the
/// monitor's own mode that macOS can be told of, else in the mode of
/// `wanted` pixels, the firmware's, if it has it: the guest's mode is not
/// set yet and follows the display taken. The GPU that showed the firmware's
/// picture, or the one GPU bound when that is unknown, has a display known
/// or likely to exist, so it is awaited within [`RETURN_SECONDS`], once.
/// What stops one GPU is reported with what its connectors detected, and
/// its node is kept for the scanout's looks.
pub(in crate::engine) fn take(
    wanted: (usize, usize),
    firmware: Shown<'_>,
    size: Option<DisplaySize>,
) {
    let Some(gpu) = super::installed() else {
        return;
    };
    let native = kms::Wanted::Native((wanted.0 as u32, wanted.1 as u32));
    let wanted = match size {
        Some(size) => match size.refresh_hz {
            Some(hz) => kms::Wanted::SizeAtRefresh((size.width, size.height), hz),
            None => kms::Wanted::Size((size.width, size.height)),
        },
        None => native,
    };
    let bound = &gpu.start.bound;
    // SAFETY: the one boot CPU, before any reader.
    let (taken, waiting) = unsafe { &mut *TAKEN.0.get() };
    for (index, bound_gpu) in bound.iter().enumerate() {
        let name = &bound_gpu.name;
        let Some(fd) = client::minor(index).and_then(client::open_node) else {
            continue;
        };
        let began = host::rdtsc();
        let awaited = match firmware {
            Shown::By(function) => function == name.as_str(),
            // Not one below a bridge that forwards none of the screen.
            Shown::Unknown => bound.len() == 1 && bound_gpu.shows_screen,
            Shown::Nothing => false,
        };
        let patience = match awaited {
            true => RETURN_SECONDS.saturating_mul(gpu.tsc_hz),
            false => 0,
        };
        let result = loop {
            match Display::take_fitting(Node(fd), wanted, &stela_soc::dcp::modes::advertisable) {
                Err(DisplayError::NoOutput) if host::rdtsc().wrapping_sub(began) < patience => {
                    let step = host::rdtsc();
                    while host::rdtsc().wrapping_sub(step) < gpu.tsc_hz / 1000 * RETURN_STEP_MILLIS
                    {
                        core::hint::spin_loop();
                    }
                }
                result => break result,
            }
        };
        let (result, uses_native) = match (result, wanted) {
            (
                Err(DisplayError::ModeUnavailable),
                kms::Wanted::SizeAtRefresh((width, height), hz),
            ) => {
                output::arguments(format_args!(
                    "GPU {name}: no output lists {width}x{height} at {hz} Hz; using the normal display mode\n"
                ));
                (
                    Display::take_fitting(Node(fd), native, &stela_soc::dcp::modes::advertisable),
                    true,
                )
            }
            (result, _) => (result, size.is_none()),
        };
        let waited_ms = host::rdtsc().wrapping_sub(began) / (gpu.tsc_hz / 1000).max(1);
        match result {
            Ok(display) => {
                if waited_ms >= RETURN_STEP_MILLIS {
                    output::arguments(format_args!(
                        "GPU {name}: display detected after {waited_ms} ms\n"
                    ));
                }
                let (width, height) = display.size();
                output::arguments(format_args!(
                    "GPU {name}: display runs {width}x{height} at {} mHz\n",
                    display.refresh_millihertz()
                ));
                if uses_native && !display.preferred_listed() {
                    output::arguments(format_args!(
                        "GPU {name}: the output lists no preferred mode (its EDID marks none, or \
                         Linux left out the monitor's own mode as past the link's clock), so it \
                         runs its first listed mode, as Linux does\n"
                    ));
                }
                report_ports(name, &display);
                // A GPU that serves no Metal device shows what another
                // draws: the display is wired to it (a laptop's panel on its
                // integrated GPU), and each frame is copied into its buffer.
                let chosen = super::gate::metal_choice(bound, Some(index));
                if !chosen.serve.contains(&index)
                    && let Some(&drawing) = chosen.serve.first()
                {
                    output::arguments(format_args!(
                        "GPU {name}: the connected display is on this GPU, so it shows the \
                         frames GPU {} draws, each copied into its buffer and flipped on its \
                         own vblank\n",
                        bound[drawing].name
                    ));
                }
                TAKEN_INDEX.store(index, Ordering::Relaxed);
                report_outputs(name, &display);
                crate::status::take_screen(Some(screen(&display)));
                for (_, fd) in waiting.drain(..) {
                    client::close_descriptor(fd);
                }
                *taken = Some((name.clone(), display));
                return;
            }
            Err(error) => {
                output::arguments(format_args!(
                    "GPU {name}: no display after {waited_ms} ms: {error:?}\n"
                ));
                match kms::survey(&mut Node(fd)) {
                    Ok(seen) => {
                        for seen in seen {
                            output::arguments(format_args!("GPU {name}: output {seen}\n"));
                        }
                    }
                    Err(error) => output::arguments(format_args!(
                        "GPU {name}: outputs unreadable: {error:?}\n"
                    )),
                }
                waiting.push((name.clone(), fd));
            }
        }
    }
}

/// The mode of the display [`take`] took, before the board's DCP takes it.
pub(in crate::engine) fn taken_size() -> Option<(usize, usize)> {
    // SAFETY: the one boot CPU, after `take` and before `scanout`.
    let taken = unsafe { &(*TAKEN.0.get()).0 };
    taken.as_ref().map(|(_, display)| {
        let (width, height) = display.size();
        (width as usize, height as usize)
    })
}

/// Give up the display [`take`] took and the nodes it kept, as amdgpu's
/// reset of their GPU left them: the reset unmapped the display's buffers
/// and holds the display's lock until it ends. Nothing is released, as a
/// release calls the driver that waits for that end, and the console stops
/// drawing on the unmapped buffer, which a draw would fault on.
pub(in crate::engine) fn abandon() {
    // SAFETY: the one boot CPU, after `take`.
    let (taken, waiting) = unsafe { &mut *TAKEN.0.get() };
    let had = taken.is_some();
    if let Some(display) = taken.take() {
        core::mem::forget(display);
    }
    let nodes = waiting.len();
    core::mem::forget(core::mem::take(waiting));
    if had && !crate::console::forget_screen() {
        output::arguments(format_args!(
            "GPU display: the console was busy as its screen was given up\n"
        ));
    }
    output::arguments(format_args!(
        "GPU display given up after the GPU's reset: display={had} nodes={nodes}\n"
    ));
}

/// The scanout of the display [`take`] took, in its own mode, for the
/// board's DCP.
pub(in crate::engine) fn scanout() -> Option<Box<dyn Scanout>> {
    let gpu = super::installed()?;
    // SAFETY: the one boot CPU, after `take`.
    let (name, display) = unsafe { (*TAKEN.0.get()).0.take() }?;
    let (width, height) = display.size();
    let mut scanout = KmsScanout::new(
        ScanoutMode {
            width,
            height,
            format: ScanoutFormat::Bgra8888,
        },
        display.refresh_millihertz(),
        display.size_mm(),
        display.edid().and_then(MonitorId::from_edid),
        gpu.tsc_hz,
    );
    scanout.display = Some((name, display));
    Some(Box::new(scanout))
}

/// Whether a GPU is bound, so it draws macOS's frames, while [`scanout`]
/// found no display taken: only the firmware's screen, on an adapter no
/// bound GPU's start ended, can show them, each frame copied.
pub(in crate::engine) fn drawn_without_display() -> bool {
    super::installed().is_some_and(|gpu| !gpu.start.bound.is_empty())
}

/// With no display taken and the firmware's `screen` gone, or with no
/// firmware screen at all ([`guest_screen::DEFAULT`]), a scanout in the
/// guest's mode that shows the first display a bound GPU detects later in
/// it. None when no GPU's node is open.
///
/// [`guest_screen::DEFAULT`]: stela_boot_core::guest_screen::DEFAULT
pub(in crate::engine) fn awaited(screen: Option<Framebuffer>) -> Option<Box<dyn Scanout>> {
    let gpu = super::installed()?;
    // SAFETY: the one boot CPU, after `take`.
    let waiting = core::mem::take(unsafe { &mut (*TAKEN.0.get()).1 });
    if waiting.is_empty() {
        return None;
    }
    use stela_boot_core::guest_screen;
    let ((width, height), refresh, size_mm, monitor) = match screen {
        Some(screen) => (
            screen.resolution(),
            screen.refresh_millihertz(),
            screen.size_mm(),
            screen.monitor(),
        ),
        None => (
            guest_screen::DEFAULT,
            guest_screen::DEFAULT_REFRESH_MILLIHERTZ,
            None,
            None,
        ),
    };
    let mut scanout = KmsScanout::new(
        ScanoutMode {
            width: width as u32,
            height: height as u32,
            format: ScanoutFormat::Bgra8888,
        },
        refresh,
        size_mm,
        monitor,
        gpu.tsc_hz,
    );
    scanout.waiting = waiting;
    output::arguments(format_args!(
        "DISPLAY no GPU display yet: looking for one in {width}x{height} once a second\n"
    ));
    Some(Box::new(scanout))
}
