import streamlit as st
from main import app, AgentState, HumanMessage, AIMessage

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="centered")

st.title("Flight Price Predictor")
st.markdown("Hello! I can help you find estimated flight prices across different airlines. Just tell me about your desired flight (e.g., 'I want an **economy class flight** from **Delhi to Mumbai**, departing **morning** with **one stop**').")

# --- Initialize Session State ---
if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {
        "messages": [AIMessage(content="What are your flight details?")], # Initial bot message
        "parsed_input": {},
        "missing_fields": [],
        "all_prices": [] 
    }
if 'chat_display_messages' not in st.session_state:
    st.session_state.chat_display_messages = [{"role": "assistant", "content": "What are your flight details?"}]

# --- Chat Interface with Scrollable Container ---
chat_container = st.container(height=400, border=True)

with chat_container:
    for message in st.session_state.chat_display_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# --- User Input Form ---
latest_assistant_message_content = ""
if st.session_state.chat_display_messages and st.session_state.chat_display_messages[-1]["role"] == "assistant":
    last_ai_content = st.session_state.chat_display_messages[-1]["content"]
    if "missing some information" in last_ai_content or "Could you please tell me" in last_ai_content or "I also need" in last_ai_content:
        latest_assistant_message_content = last_ai_content

with st.form(key="user_input_form", clear_on_submit=True):
    user_input = st.text_input(
        "Your message:",
        placeholder=latest_assistant_message_content or "Type your flight details here...",
        key="user_input_text"
    )
    col1, col2 = st.columns([2, 9])
    with col1:
        send_button = st.form_submit_button("Send")
    with col2:
        reset_button_form = st.form_submit_button("Start Over")


# --- Process User Input (on Send or Reset) ---
if send_button and user_input:
    st.session_state.agent_state["messages"].append(HumanMessage(content=user_input))
    st.session_state.chat_display_messages.append({"role": "user", "content": user_input})

    with chat_container:
        with st.chat_message("user"):
            st.write(user_input)

    try:
        final_state = app.invoke(st.session_state.agent_state)
        st.session_state.agent_state = final_state 

        last_ai_message = next((msg for msg in reversed(final_state["messages"]) if isinstance(msg, AIMessage)), None)
        
        if last_ai_message:
            ai_response_content = last_ai_message.content
            st.session_state.chat_display_messages.append({"role": "assistant", "content": ai_response_content})
            with chat_container:
                with st.chat_message("assistant"):
                    st.write(ai_response_content)

            if "Here are the top 3 lowest estimated flight prices" in ai_response_content and st.session_state.agent_state["all_prices"]:
                st.info("Flight price comparison complete! You can now start a new query or refine your previous one.")
                st.session_state.agent_state = {
                    "messages": [], 
                    "parsed_input": {},
                    "missing_fields": [],
                    "all_prices": []
                }
                st.session_state.chat_display_messages.append({"role": "assistant", "content": "What's your next flight query?"})

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}. Please try again or click 'Start Over'."
        st.session_state.chat_display_messages.append({"role": "assistant", "content": error_message})
        with chat_container:
            with st.chat_message("assistant"):
                st.error(error_message)
        st.session_state.agent_state = { 
            "messages": [],
            "parsed_input": {},
            "missing_fields": [],
            "all_prices": []
        }

# --- Handle Reset Button (outside the form if needed, or inside for clarity) ---
if reset_button_form: 
    st.session_state.agent_state = {
        "messages": [],
        "parsed_input": {},
        "missing_fields": [],
        "all_prices": []
    }
    st.session_state.chat_display_messages = [{"role": "assistant", "content": "Welcome! How can I help you find flight prices today?"}]
    st.rerun() 


# Optional: Display current state for debugging
with st.expander("Debug Information"):
    st.write("--- Current LangGraph AgentState ---")
    st.json(st.session_state.agent_state)
    st.write("--- Displayed Chat Messages ---")
    st.json(st.session_state.chat_display_messages)