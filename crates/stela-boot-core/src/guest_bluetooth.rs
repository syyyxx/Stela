//! The boot log's one line about the guest's Bluetooth. A boot with host
//! input gives macOS a model of Apple's internal USB Bluetooth module
//! (`stela_soc::usb::hci::model`): BlueTool's chip boot matches it and
//! bluetoothd brings it up, so the daemon stays up and Setup's first screen
//! (Language Chooser), which asks it for the paired devices before it
//! draws, gets an empty list at once.
//!
//! The model has no radio. It is powered, has a stable locally administered
//! address, accepts scans and connection attempts and finds nothing, and has
//! no paired devices. That is a degraded answer, not Bluetooth, and the line
//! says so once. A host controller is not lent to the guest.

/// The line for a boot that gives macOS host input, by whether the host
/// has a Bluetooth controller; `None` for a boot without host input.
pub fn model_line(host_controller: bool, input: bool) -> Option<&'static str> {
    if !input {
        return None;
    }
    Some(if host_controller {
        "BLUETOOTH MODEL the guest's controller is a model with no radio: it is powered, \
         finds no devices and has none paired; this computer's controller is not lent to it\n"
    } else {
        "BLUETOOTH MODEL the guest's controller is a model with no radio: it is powered, \
         finds no devices and has none paired\n"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_boot_with_input_says_once_that_the_controller_has_no_radio() {
        for host_controller in [false, true] {
            let line = model_line(host_controller, true).expect("reported");
            assert!(line.starts_with("BLUETOOTH MODEL the guest's controller is a model "));
            assert!(line.contains("no radio"));
            assert!(line.contains("finds no devices and has none paired"));
            assert!(line.ends_with('\n') && line.matches('\n').count() == 1);
        }
    }

    #[test]
    fn a_host_controller_is_named_as_not_lent_and_a_boot_without_input_says_nothing() {
        let line = model_line(true, true).expect("reported");
        assert!(line.contains("controller is not lent"));
        assert!(!model_line(false, true).unwrap().contains("lent"));
        assert_eq!(model_line(true, false), None);
        assert_eq!(model_line(false, false), None);
    }
}
