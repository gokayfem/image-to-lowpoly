# Claude (claude.ai Projects)

Prefer the skill: zip the `skills/image-to-lowpoly/` folder and upload it under **Settings > Capabilities > Skills**.
Claude then loads `SKILL.md`, the references and the tools on demand. Its code environment has no Blender, so
it can plan allocations, write build scripts, read your renders and check counts in `mesh.json` and `.glb`
files; you run the Blender stages locally.

Without skills, use a Project:

1. Paste `adapters/universal-system-prompt.md` (the part below the line) into the project instructions.
2. Add to project knowledge: `SKILL.md`, `references/counting.md`, `references/budget-worker.md` and
   `references/lessons.md`.

For Claude Code, install the plugin or copy the skill folder (see the repository README). Claude Code runs the
scripts and Blender itself, looks at every render, and starts the budget workers and the blind reviewer as
subagents.
