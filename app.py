import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ Google API Key not found. Please set it in your environment variables or a .env file.")
    st.stop()

try:
    genai.configure(api_key="AIzaSyBjlAL2DMbeDQl7DrrMeFtP5oe1jBiWctA")
    model = genai.GenerativeModel('gemini-1.5-flash') # Or your preferred model
except Exception as e:
    st.error(f"❌ Error configuring Gemini API: {e}")
    st.stop()

# --- System Prompts ---

# Adjusted slightly to hint towards scaled assessment
TRUST_SYSTEM_PROMPT = """
You are an AI assistant skilled in analyzing conversations to identify potential areas of ambiguity, misunderstanding, or points needing clarification to build trust before formalizing an agreement. Based on the following conversation transcript, generate 4-5 specific questions designed to probe these areas. The questions should help the parties gauge confidence, clarity, or agreement on key topics. Frame the questions neutrally so they can be assessed on a scale (e.g., regarding certainty, completeness, fairness). Output *only* the questions, each on a new line, numbered or bulleted. Do not add any introductory or concluding text.

Conversation:
{conversation_text}
"""

# Updated to interpret 1-5 scale
TRUST_SUMMARIZATION_PROMPT = """
You are an AI assistant evaluating the level of trust between parties based on answers to specific questions, rated on a 1-5 scale. Analyze the following questions and their corresponding numerical answers. The scale represents: 1 = Very Low (Confidence/Clarity/Agreement), 2 = Low, 3 = Medium, 4 = High, 5 = Very High.

Based *only* on these scaled answers, determine the overall trust level. Consider the average score, the presence of low scores, and the importance of the topics addressed by the questions. Output *only* one of the following three levels: High, Medium, or Low. Do not provide any explanation or justification, just the single word category.

Questions and Scaled Answers (1-5):
{questions_and_answers}
"""

# Drafting prompt remains the same

DRAFTING_SYSTEM_PROMPT = """
You are an AI legal assistant specializing in drafting preliminary contract outlines based on informal conversations. You will be provided with:
1.  The original conversation transcript.
2.  An assessed trust level ('High', 'Medium', or 'Low') derived from user answers to specific trust-related questions about the conversation.

Your task is to generate a basic contract draft that accurately reflects the key points and specific details agreed upon in the conversation. Adapt the tone and specific clauses based on the provided trust level:

*   **If Trust Level is High:** Generate a standard, collaborative draft focusing on clearly outlining the core agreement (scope, deliverables, payment, timeline) with standard, balanced clauses. Assume good faith and keep protective clauses minimal and reciprocal.
*   **If Trust Level is Medium:** Generate a draft that includes clearer definitions, specific checkpoints or milestones, more detailed payment terms (e.g., tied to deliverables), and standard dispute resolution options. Add clauses that encourage transparency and verification without being overly adversarial.
*   **If Trust Level is Low:** Generate a draft with stricter, more protective clauses. Emphasize verification mechanisms, potentially require upfront payment or escrow, include specific penalties for delays or non-performance, define breach conditions clearly, and suggest more formal dispute resolution methods (e.g., arbitration). Ensure all obligations and potential risks are explicitly addressed.

**CRITICAL Instructions:**
*   **EXTRACT & INSERT DETAILS:** Carefully parse the **Conversation Transcript** provided below. Identify the **specific names** of the parties involved (e.g., individuals, companies), the exact **numerical values** discussed (e.g., loan amounts like $5000, payment figures, interest rates like 6%), specific **dates** or **timelines** mentioned (e.g., "within 6 months", "by July 1st"), and concrete descriptions of **scope/deliverables**. You **MUST** insert these exact extracted details directly into the relevant sections and clauses of the contract draft (e.g., Parties section, Loan Amount clause, Interest Rate clause, Repayment Term clause).
*   **PRIORITIZE EXTRACTED INFO:** Base the core substance AND the specific details (names, amounts, dates, scope) of the contract *directly* on the information found within the **Conversation Transcript**.
*   **USE PLACEHOLDERS ONLY IF NECESSARY:** Use generic placeholders like `[Party A Name]`, `[Party B Name]`, `[Date]`, `[Address]`, `[State/Jurisdiction]`, `[Specific Detail Placeholder]` *only* for standard contract information that is clearly *absent* from the provided **Conversation Transcript** (like full addresses or a governing law jurisdiction if not mentioned). **Do NOT use placeholders for names, amounts, or key terms if they ARE mentioned in the transcript.**
*   **Structure:** Structure the draft logically (e.g., Parties (using extracted names), Background/Recitals (if applicable from context), Core Agreement (Loan Amount, Interest Rate, Scope of Work - using extracted details), Payment Terms (using extracted details), Term/Duration (using extracted details), Default, Confidentiality (if applicable), Warranties/Disclaimers (adjust based on trust), Dispute Resolution (adjust based on trust), Governing Law (use placeholder if not mentioned), Signatures).
*   **Language:** Use clear and concise language. Avoid excessive legal jargon but maintain a professional tone.
*   **Goal:** Focus on creating a functional first draft or outline reflecting the conversation, not a fully exhaustive legal document ready for signature without review.

Begin the draft now, meticulously incorporating the specific details from the conversation.

**Conversation Transcript:**
{conversation_text}

**Assessed Trust Level:**
{trust_level}
"""


# --- Helper Function ---
def call_gemini_api(prompt, retries=2):
    """Sends a prompt to the Gemini API and returns the response."""
    try:
        response = model.generate_content(prompt)
        if not response.parts:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                st.error(f"API call blocked. Reason: {response.prompt_feedback.block_reason}")
                return None
            else:
                st.warning("API returned an empty response. Retrying...")
                if retries > 0:
                    return call_gemini_api(prompt, retries - 1)
                else:
                    st.error("API returned an empty response after multiple retries.")
                    return None
        return response.text
    except Exception as e:
        st.error(f"An error occurred during the API call: {e}")
        return None

def parse_questions(text):
    """Parses numbered or bulleted questions from text."""
    lines = text.strip().split('\n')
    questions = [re.sub(r"^\s*[\d\.\)\-\*]+\s*", "", line).strip() for line in lines if re.match(r"^\s*[\d\.\)\-\*]+\s+", line)]
    if not questions:
        questions = [line.strip() for line in lines if line.strip()]
    return questions


# --- Initialize Session State ---
DEFAULT_TRUST_SCORE = 3 # Default neutral score
if 'app_stage' not in st.session_state:
    st.session_state.app_stage = "input"
if 'conversation' not in st.session_state:
    st.session_state.conversation = ""
if 'trust_questions' not in st.session_state:
    st.session_state.trust_questions = []
if 'trust_answers' not in st.session_state:
    st.session_state.trust_answers = {} # Use dict for answers keyed by question index, will store numbers
if 'trust_level' not in st.session_state:
    st.session_state.trust_level = None
if 'contract_draft' not in st.session_state:
    st.session_state.contract_draft = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# --- App UI and Logic ---

st.set_page_config(layout="wide")
st.title("📝 Conversation to Contract Draft Assistant")
st.caption("Analyzes conversation, assesses trust via scaled questions, and drafts a preliminary contract.")

# --- Stage 1: Input Conversation ---
if st.session_state.app_stage == "input":
    st.header("1. Input Conversation Text")
    conversation_input = st.text_area("Paste the conversation transcript here:", height=200, key="conv_input_main", value=st.session_state.conversation)
    st.session_state.conversation = conversation_input

    if st.button("🔍 Generate Trust Questions", key="generate_q_button"):
        if not st.session_state.conversation:
            st.warning("⚠️ Please paste the conversation text first.")
            st.session_state.error_message = "Conversation text is missing."
        else:
            st.session_state.error_message = None
            prompt = TRUST_SYSTEM_PROMPT.format(conversation_text=st.session_state.conversation)
            with st.spinner("🤔 Analyzing conversation and generating questions..."):
                response_text = call_gemini_api(prompt)

            if response_text:
                st.session_state.trust_questions = parse_questions(response_text)
                if not st.session_state.trust_questions:
                     st.error("❌ Could not parse questions from the API response. The response was:")
                     st.code(response_text)
                     st.session_state.error_message = "Failed to parse questions."
                else:
                    # Initialize answers with the default numerical score
                    st.session_state.trust_answers = {i: DEFAULT_TRUST_SCORE for i in range(len(st.session_state.trust_questions))}
                    st.session_state.app_stage = "questions"
                    st.rerun()
            else:
                st.error("❌ Failed to generate trust questions from the API.")
                st.session_state.error_message = "API call for questions failed."

    if st.session_state.error_message:
        st.error(st.session_state.error_message)


# --- Stage 2: Answer Trust Questions (Scaled) ---
elif st.session_state.app_stage == "questions":
    st.header("2. Assess Trust on a Scale (1-5)")

    with st.expander("Original Conversation", expanded=False):
        st.markdown(f"```\n{st.session_state.conversation}\n```")

    st.subheader("Rate your confidence/clarity/agreement for each point:")
    st.markdown("_(Scale: 1 = Very Low, 2 = Low, 3 = Medium, 4 = High, 5 = Very High)_")

    answers_temp = {} # Store answers temporarily

    if not st.session_state.trust_questions:
         st.warning("⚠️ No trust questions were generated. Please go back and try again.")
         if st.button("Go Back"):
             st.session_state.app_stage = "input"
             st.rerun()
    else:
        # Use a form to collect all slider inputs before processing
        with st.form(key='trust_answers_form'):
            for i, question in enumerate(st.session_state.trust_questions):
                # Use select_slider for the 1-5 scale
                answer = st.select_slider(
                    f"**Q{i+1}: {question}**",
                    options=[1, 2, 3, 4, 5],
                    value=st.session_state.trust_answers.get(i, DEFAULT_TRUST_SCORE), # Get current or default score
                    key=f"answer_slider_{i}",
                    help="Rate your confidence, clarity, or agreement regarding this point (1=Very Low, 5=Very High)."
                )
                answers_temp[i] = answer # Collect current answer from slider

            submitted = st.form_submit_button("✅ Calculate Trust & Generate Draft")

            if submitted:
                st.session_state.trust_answers = answers_temp # Store collected answers in session state
                st.session_state.error_message = None # Clear previous errors

                # Prepare Q&A string for the summarization prompt
                q_and_a_string = ""
                for i, question in enumerate(st.session_state.trust_questions):
                    # Format with the numerical score
                    q_and_a_string += f"Question {i+1}: {question}\nAnswer Score (1-5): {st.session_state.trust_answers[i]}\n\n"

                summary_prompt = TRUST_SUMMARIZATION_PROMPT.format(questions_and_answers=q_and_a_string.strip())

                with st.spinner("⚖️ Calculating trust level based on scores..."):
                    trust_level_response = call_gemini_api(summary_prompt)

                if trust_level_response and trust_level_response.strip() in ["High", "Medium", "Low"]:
                    st.session_state.trust_level = trust_level_response.strip()
                    st.session_state.app_stage = "drafting"
                    st.rerun() # Move to drafting stage
                else:
                    st.error(f"❌ Failed to determine trust level from API. Response: '{trust_level_response}'. Expected 'High', 'Medium', or 'Low'.")
                    st.session_state.error_message = "Trust level calculation failed."
                    st.rerun() # Rerun to show the error within the form context if needed

    # Display error message if it occurred during the submission process
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    if st.button("⬅️ Back to Conversation Input", key="back_to_input"):
         st.session_state.app_stage = "input"
         st.rerun()


# --- Stage 3: Display Draft ---
elif st.session_state.app_stage == "drafting":
    st.header("3. Generated Contract Draft")

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Original Conversation Context", expanded=False):
            st.markdown(f"```\n{st.session_state.conversation}\n```")
    with col2:
         with st.expander("Trust Q&A (Scaled Scores) Context", expanded=False):
            st.markdown("_Scale: 1 = Very Low, 2 = Low, 3 = Medium, 4 = High, 5 = Very High_")
            st.divider()
            for i, question in enumerate(st.session_state.trust_questions):
                 st.markdown(f"**Q{i+1}:** {question}")
                 # Display the numerical score stored
                 st.markdown(f"**A{i+1} Score:** {st.session_state.trust_answers.get(i, 'N/A')}")
                 st.divider()


    if st.session_state.trust_level:
        color = "green" if st.session_state.trust_level == "High" else ("orange" if st.session_state.trust_level == "Medium" else "red")
        st.info(f"**Assessed Trust Level:** :{color}[{st.session_state.trust_level}] (Based on scaled answers)")

    # Generate draft only if it hasn't been generated yet
    if not st.session_state.contract_draft:
        if st.session_state.conversation and st.session_state.trust_level:
            drafting_prompt_filled = DRAFTING_SYSTEM_PROMPT.format(
                conversation_text=st.session_state.conversation,
                trust_level=st.session_state.trust_level
            )
            with st.spinner("✍️ Generating contract draft based on conversation and trust level..."):
                draft_response = call_gemini_api(drafting_prompt_filled)

            if draft_response:
                st.session_state.contract_draft = draft_response
            else:
                st.error("❌ Failed to generate contract draft from the API.")
                st.session_state.error_message = "Draft generation API call failed."
        else:
            st.error("❌ Missing conversation or trust level. Cannot generate draft.")
            st.session_state.error_message = "Missing data for draft generation."

    # Display the generated draft
    if st.session_state.contract_draft:
        st.subheader("Draft Text:")
        st.markdown(st.session_state.contract_draft)
    elif st.session_state.error_message:
         st.error(st.session_state.error_message)


    if st.button("🔄 Start Over", key="start_over_button"):
        st.session_state.app_stage = "input"
        st.session_state.conversation = ""
        st.session_state.trust_questions = []
        st.session_state.trust_answers = {}
        st.session_state.trust_level = None
        st.session_state.contract_draft = None
        st.session_state.error_message = None
        st.rerun()