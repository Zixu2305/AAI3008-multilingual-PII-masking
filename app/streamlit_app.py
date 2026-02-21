"""Minimal Streamlit placeholder app for future demo UI."""


def main() -> None:
    try:
        import streamlit as st
    except Exception:
        print("Streamlit is not installed. Install it to run the demo app.")
        return

    st.set_page_config(page_title="PII Masking Demo", page_icon="🔒")
    st.title("Multilingual PII Masking Demo")
    st.write("This is a placeholder UI. Use scripts/ or src.cli for pipeline execution.")


if __name__ == "__main__":
    main()
