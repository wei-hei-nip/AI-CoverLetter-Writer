import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyMuPDFLoader

# Personal Input
cv_path = "PATH_TO_CV"
model = "MODEL_NAME"

# Load CV
def load_cv_text(cv_path):
    loader = PyMuPDFLoader(cv_path)
    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)

cv_text = load_cv_text(cv_path)

# Load LLM Model
llm = OllamaLLM(model=model)

# Prompt templates

base_prompt_template = """
You are a professional career assistant.

Here is the user's CV:
{cv}

Here is the job description:
{job}

Write a tailored cover letter highlighting the candidate’s relevant experience and skills.
"""

customize_prompt_template = """
You are an expert editor. Here is the original cover letter:

{letter}

Your task: {instruction}

Return the revised version only.
"""

base_prompt = PromptTemplate(
    input_variables=["cv", "job"],
    template=base_prompt_template
)

customize_prompt = PromptTemplate(
    input_variables=["letter", "instruction"],
    template=customize_prompt_template
)

# Streamlit Interface
st.title("📝 AI Cover Letter Generator with Customization")

job_description = st.text_area("Paste the job description", height=250)

if st.button("Generate Base Cover Letter"):
    if not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        with st.spinner("Generating..."):
            full_prompt = base_prompt.format(cv=cv_text, job=job_description.strip())
            base_result = llm.invoke(full_prompt)
            st.session_state.base_letter = base_result.strip()

# Show base letter if available
if "base_letter" in st.session_state:
    st.subheader("Generated Cover Letter")
    st.text_area("Base Letter", value=st.session_state.base_letter, height=350)

    # Further customise
    st.markdown("### 🎨 Customize Your Letter")
    custom_instruction = st.text_area("Enter how you want to modify the letter (e.g., 'Make it more concise and friendly')")

    if st.button("Apply Customization"):
        if not custom_instruction.strip():
            st.warning("Please provide a customization instruction.")
        else:
            with st.spinner("Applying customization..."):
                final_prompt = customize_prompt.format(
                    letter=st.session_state.base_letter,
                    instruction=custom_instruction.strip()
                )
                customized_result = llm.invoke(final_prompt)
                st.text_area("Customized Letter", value=customized_result.strip(), height=350)
