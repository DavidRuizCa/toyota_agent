import streamlit as st
from dotenv import load_dotenv
from agent.agent import Agent

# Load environment variables
load_dotenv()

# Configure the Streamlit page settings and title
st.set_page_config(page_title="Toyota Agent", layout="wide")
st.title("Toyota Agentic Assistant")

# Initialize Agent if it doesn't exist in session state
# This ensures the agent is created only once
if "agent" not in st.session_state:
    st.session_state['agent'] = Agent()

# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state['messages'] = []

# Display chat messages from history on app rerun
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # Display tool execution details if available
    if "tools" in message:
        with st.expander("Tool Details"):
            st.code(message["tools"])

# Capture user input
if prompt := st.chat_input("What would you like to know?"):
    
    # Display user message and add it to chat history
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state['messages'].append({"role": "user", "content": prompt})

    # Display assistant response 
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get response from the agent
                response = st.session_state['agent'].answer(prompt)            
                st.markdown(response["answer"])

                # Display tool details if present
                if "tools" in response:
                    with st.expander("Tool Details"):
                        st.code(response["tools"])
                    
                    # Save response to history
                    st.session_state['messages'].append({
                        "role": "assistant", 
                        "content": response["answer"],
                        "tools": response["tools"]
                    })
                else:
                    # Save response without tool details to history
                    st.session_state['messages'].append({
                        "role": "assistant", 
                        "content": response["answer"]
                    })
                
                # Rerun to update the UI with the new state
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {e}")