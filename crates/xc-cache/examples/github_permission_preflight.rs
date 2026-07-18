use std::error::Error;
use xc_cache::GitHubCredentialApiProbe;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args();
    let program = arguments
        .next()
        .unwrap_or_else(|| "github_permission_preflight".to_owned());
    let Some(repository) = arguments.next() else {
        eprintln!("usage: {program} OWNER/REPOSITORY");
        std::process::exit(2);
    };
    if arguments.next().is_some() {
        eprintln!("usage: {program} OWNER/REPOSITORY");
        std::process::exit(2);
    }

    let session = GitHubCredentialApiProbe::default().probe_repository(&repository)?;
    println!("{}", serde_json::to_string_pretty(session.evidence())?);
    Ok(())
}
