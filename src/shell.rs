use hermit_sync::Lazy;
use simple_shell::*;

use crate::arch::kernel::COM1;
use crate::interrupts::print_statistics;

fn read() -> Option<u8> {
	COM1.lock().as_mut().map(|s| s.read())?
}

static mut SHELL: Lazy<Shell<'_>> = Lazy::new(|| {
	let (print, read) = (|s: &str| print!("{}", s), read);
	let mut shell = Shell::new(print, read);

	shell.commands.insert(
		"help",
		ShellCommand {
			help: "Print this help message",
			func: |_, shell| {
				shell.print_help_screen();
				Ok(())
			},
			aliases: &["?", "h"],
		},
	);
	shell.commands.insert(
		"interrupts",
		ShellCommand {
			help: "Shows the number of IRQs",
			func: |_, shell| {
				print_statistics();
				Ok(())
			},
			aliases: &["i"],
		},
	);
	shell.commands.insert(
		"shutdown",
		ShellCommand {
			help: "Shutdown HermitOS",
			func: |_, shell| {
				crate::__sys_shutdown(0);
				Ok(())
			},
			aliases: &["s"],
		},
	);
	#[cfg(any(feature = "tcp", feature = "udp"))]
	shell.commands.insert(
		"ip",
		ShellCommand {
			help: "Show network config",
			func: |_, shell| {
				crate::executor::network::print_network_configuration();
				Ok(())
			},
			aliases: &["ip"],
		},
	);

	shell
});

pub(crate) fn init() {
	crate::executor::spawn(unsafe { SHELL.run_async() });
}
