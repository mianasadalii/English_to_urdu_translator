# ============================================================
# English ↔ Urdu Translator — app.py
# Built with HuggingFace Transformers + Gradio
# ============================================================

# --- Step 1: Import required libraries ---
import gradio as gr
from transformers import MarianMTModel, MarianTokenizer

# ============================================================
# Step 2: Load models ONCE at startup (not on every request)
# This makes the app fast after the first load.
# ============================================================

print("⏳ Loading English → Urdu model...")
EN_UR_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ur"
en_ur_tokenizer = MarianTokenizer.from_pretrained(EN_UR_MODEL_NAME)
en_ur_model     = MarianMTModel.from_pretrained(EN_UR_MODEL_NAME)
print("✅ English → Urdu model ready.")

print("⏳ Loading Urdu → English model...")
UR_EN_MODEL_NAME = "Helsinki-NLP/opus-mt-ur-en"
ur_en_tokenizer = MarianTokenizer.from_pretrained(UR_EN_MODEL_NAME)
ur_en_model     = MarianMTModel.from_pretrained(UR_EN_MODEL_NAME)
print("✅ Urdu → English model ready.")


# ============================================================
# Step 3: Core translation function
# direction: "en→ur"  or  "ur→en"
# ============================================================

def translate(text: str, direction: str) -> str:
    """
    Translate text between English and Urdu.
    Args:
        text      : The source text entered by the user.
        direction : "en→ur" for English→Urdu, "ur→en" for Urdu→English.
    Returns:
        Translated string, or an error/warning message.
    """

    # --- Guard: empty input ---
    if not text or not text.strip():
        return "⚠️  Please enter some text before translating."

    # --- Pick the right model & tokenizer ---
    if direction == "en→ur":
        tokenizer = en_ur_tokenizer
        model     = en_ur_model
    else:
        tokenizer = ur_en_tokenizer
        model     = ur_en_model

    try:
        # Tokenize: convert text to numbers the model understands
        inputs = tokenizer(
            text,
            return_tensors="pt",   # PyTorch tensors
            padding=True,
            truncation=True,
            max_length=512         # safety limit
        )

        # Generate translation tokens
        translated_tokens = model.generate(**inputs)

        # Decode tokens back to a human-readable string
        translation = tokenizer.decode(
            translated_tokens[0],
            skip_special_tokens=True   # remove <pad>, </s>, etc.
        )

        return translation

    except Exception as e:
        # Catch anything unexpected and show a friendly message
        return f"❌ Translation error: {str(e)}"


# ============================================================
# Step 4: Build the Gradio UI
# ============================================================

# Custom CSS for a polished, readable layout
custom_css = """
/* ---------- page background ---------- */
body, .gradio-container {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
    font-family: 'Segoe UI', sans-serif;
}
/* ---------- card / panel ---------- */
.gr-box, .gr-panel, .gr-form {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px);
}
/* ---------- labels ---------- */
label span, .gr-block label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
/* ---------- text areas ---------- */
textarea {
    background: rgba(0,0,0,0.35) !important;
    color: #f0f4f8 !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
}
/* ---------- translate button ---------- */
#translate-btn {
    background: linear-gradient(90deg, #4facfe, #00f2fe) !important;
    color: #0f2027 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
#translate-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79,172,254,0.45) !important;
}
/* ---------- radio buttons ---------- */
.gr-radio label { color: #cbd5e0 !important; }
/* ---------- title / description ---------- */
h1 { color: #ffffff !important; }
.description { color: #94a3b8 !important; }
"""

with gr.Blocks(title="English ↔ Urdu Translator") as demo:

    # ----- Header -----
    gr.HTML("""
    <div style="text-align:center; padding: 24px 0 8px;">
        <h1 style="font-size:2.4rem; font-weight:800; letter-spacing:-0.5px; margin:0;">
            🌐 English ↔ Urdu Translator
        </h1>
        <p class="description" style="font-size:1.05rem; margin-top:10px;">
            Powered by Helsinki-NLP MarianMT · Built with 🤗 Transformers + Gradio
        </p>
    </div>
    """)

    # ----- Direction toggle -----
    direction = gr.Radio(
        choices=["en→ur", "ur→en"],
        value="en→ur",
        label="Translation Direction",
        info="Choose which language to translate from"
    )

    # ----- Input / Output row -----
    with gr.Row():
        input_text = gr.Textbox(
            lines=6,
            placeholder="Type your English text here…",
            label="📝 Source Text"
        )
        output_text = gr.Textbox(
            lines=6,
            placeholder="Translation will appear here…",
            label="✨ Translation",
            interactive=False    # read-only output
        )

    # Update placeholder dynamically when direction changes
    def update_placeholder(dir_val):
        if dir_val == "en→ur":
            return gr.update(placeholder="Type your English text here…",  label="📝 English Source")
        else:
            return gr.update(placeholder="یہاں اردو متن لکھیں…", label="📝 Urdu Source")

    direction.change(fn=update_placeholder, inputs=direction, outputs=input_text)

    # ----- Translate button -----
    translate_btn = gr.Button("Translate ➜", elem_id="translate-btn", variant="primary")

    # Wire the button to the translate function
    translate_btn.click(
        fn=translate,
        inputs=[input_text, direction],
        outputs=output_text
    )

    # Also allow pressing Enter (submit on textbox)
    input_text.submit(
        fn=translate,
        inputs=[input_text, direction],
        outputs=output_text
    )

    # ----- Examples -----
    gr.Examples(
        examples=[
            ["Hello, how are you?",                     "en→ur"],
            ["Pakistan is a beautiful country.",         "en→ur"],
            ["Artificial intelligence is the future.",   "en→ur"],
            ["آپ کیسے ہیں؟",                            "ur→en"],
            ["پاکستان ایک خوبصورت ملک ہے۔",             "ur→en"],
        ],
        inputs=[input_text, direction],
        outputs=output_text,
        fn=translate,
        cache_examples=False,
        label="📚 Try these examples"
    )

    # ----- Footer -----
    gr.HTML("""
    <div style="text-align:center; padding:20px 0 4px; color:#64748b; font-size:0.85rem;">
        Models: Helsinki-NLP/opus-mt-en-ur &amp; Helsinki-NLP/opus-mt-ur-en &nbsp;|&nbsp;
        Made with ❤️ using HuggingFace + Gradio
    </div>
    """)


# ============================================================
# Step 5: Launch the app
# - share=True  → creates a public link (useful in Colab)
# - For HF Spaces, just call demo.launch() with no arguments
# ============================================================

if __name__ == "__main__":
    demo.launch(css=custom_css)
