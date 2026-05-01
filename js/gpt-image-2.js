// Frontend tweak for the comfyui-gpt-image-2 nodes.
// Makes the `prompt` textbox visibly larger on first render so long prompts
// don't get hidden behind a one-line widget. Users can still drag-resize.

import { app } from "../../scripts/app.js";

const TARGET_NODES = new Set([
    "GPTImage2Generate",
    "GPTImage2Edit",
    "GPTImage2EstimateCost",
]);

const PROMPT_HEIGHT = 220;   // pixels for the textarea itself
const NODE_MIN_HEIGHT = 520; // pixels for the whole node

app.registerExtension({
    name: "comfyui.gpt-image-2.ui",

    nodeCreated(node) {
        if (!TARGET_NODES.has(node.comfyClass)) return;

        const promptWidget = node.widgets?.find((w) => w.name === "prompt");
        if (!promptWidget) return;

        // Hint to ComfyUI's layout engine that this widget wants more vertical
        // space. `computedHeight` is the property the multiline widget reads
        // when it computes its drawn size.
        promptWidget.computedHeight = PROMPT_HEIGHT;

        // Bump the node itself so the widget actually has somewhere to render.
        const [w, h] = node.size;
        if (h < NODE_MIN_HEIGHT) {
            node.size = [Math.max(w, 420), NODE_MIN_HEIGHT];
        }

        // Some Comfy frontend versions expose the underlying <textarea> via
        // `inputEl`. If present, set its rows attribute too — purely cosmetic
        // when the layout engine ignores `computedHeight`.
        if (promptWidget.inputEl) {
            promptWidget.inputEl.rows = 10;
            promptWidget.inputEl.style.minHeight = `${PROMPT_HEIGHT}px`;
        }
    },
});
