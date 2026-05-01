use std::fs;

const AGENT_NAMES: &[&str] = &[
    "opencode",
    "codex",
    "claude-code",
    "claude",
    "aider",
    "cursor",
    "windsurf",
    "github-copilot",
    "copilot",
    "tabby",
    "continue",
    "junior",
];

pub fn has_ai_agent_ancestor() -> bool {
    let mut pid = std::process::id() as i32;

    while pid > 1 {
        let status_path = format!("/proc/{pid}/status");
        let content = match fs::read_to_string(&status_path) {
            Ok(c) => c,
            Err(_) => break,
        };

        let mut name = None;
        let mut ppid = None;

        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("Name:\t") {
                name = Some(rest.trim());
            } else if let Some(rest) = line.strip_prefix("PPid:\t") {
                ppid = rest.trim().parse::<i32>().ok();
            }
        }

        if let Some(name) = name {
            if AGENT_NAMES.iter().any(|a| name.eq_ignore_ascii_case(a)) {
                return true;
            }
        }

        match ppid {
            Some(p) if p > 0 && p != pid => pid = p,
            _ => break,
        }
    }

    false
}
