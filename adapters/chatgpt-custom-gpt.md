# ChatGPT (Custom GPT)

1. Create a GPT and paste `adapters/universal-system-prompt.md` (the part below the line) into **Instructions**.
2. Upload as **Knowledge**: `skills/image-to-lowpoly/SKILL.md` and the files in `skills/image-to-lowpoly/references/`.
3. Enable **Code Interpreter** so it can count triangles in uploaded `mesh.json` and `.glb` files and run the
   plain-Python parts of `scripts/catlib/` (topology edits, the CPU rasteriser). Blender and the renders run on
   your machine.
