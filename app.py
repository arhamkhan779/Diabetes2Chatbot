import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
import logging
import hashlib
from datetime import datetime
load_dotenv()
import os

# =========================================================
# Configuration
# =========================================================

# api_key = st.secrets["GEMINI_API_KEY"]

# Import database functions
from database_module import (
    init_db, create_user, get_user_by_email, get_user_by_id,
    save_user_profile, get_user_profile, is_profile_completed,
    add_insulin, get_insulin_schedule, delete_insulin_by_user,
    add_medication, get_medications, delete_medications_by_user,
    save_chat_message, get_chat_history, clear_chat_history,
    build_patient_context
)

# load_dotenv()

# # =========================================================
# # Configuration
# # =========================================================

api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# Load FAISS index and text chunks
faiss_index = faiss.read_index(r"faq_index.faiss")
with open(r"faq_texts.pkl", "rb") as f:
    text_chunks = pickle.load(f)

model = genai.GenerativeModel("gemini-2.5-flash-lite")

# Initialize database
init_db()

# =========================================================
# Helper Functions
# =========================================================

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email: str) -> bool:
    """Basic email validation"""
    return "@" in email and "." in email.split("@")[1]

def generate_embeddings(text: str):
    """Generate embeddings using Gemini embedding model"""
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        if response and "embedding" in response:
            return response["embedding"]
        else:
            logging.warning("⚠️ No embedding returned for text.")
            return []
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        return []

def find_similar_texts(query: str, top_k=5):
    """Find similar chunks using FAISS"""
    query_embedding = np.array(generate_embeddings(query)).astype("float32").reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [text_chunks[idx].page_content for idx in indices[0] if idx < len(text_chunks)]

def format_patient_context(context: dict) -> str:
    """Format patient context for LLM prompt"""
    profile = context.get("profile", {})
    insulin = context.get("insulin_schedule", [])
    medications = context.get("medications", [])
    
    formatted = "### Patient Information:\n"
    
    if profile:
        formatted += f"""
**Demographics:**
- Age: {profile.get('age', 'N/A')} years
- Gender: {profile.get('gender', 'N/A')}
- Weight: {profile.get('weight', 'N/A')} kg
- Height: {profile.get('height', 'N/A')} cm
- Diabetes Type: {profile.get('diabetes_type', 'N/A')}
- Diagnosed in: {profile.get('diagnosis_year', 'N/A')}
- HbA1c: {profile.get('hba1c', 'N/A')}%
"""
    
    if insulin:
        formatted += "\n**Insulin Schedule:**\n"
        for idx, ins in enumerate(insulin, 1):
            formatted += f"{idx}. {ins['insulin_name']} - {ins['dosage']} ({ins['timing']})\n"
    
    if medications:
        formatted += "\n**Medications:**\n"
        for idx, med in enumerate(medications, 1):
            formatted += f"{idx}. {med['medicine_name']} - {med['dosage']} ({med['frequency']})\n"
    
    return formatted

def generate_response(query: str, user_id: str):
    """Generate personalized response based on query and patient context"""
    try:
        # Get guideline context from FAISS
        guideline_context = find_similar_texts(query)
        
        # Get patient context from database
        patient_context = build_patient_context(user_id)
        patient_info = format_patient_context(patient_context)
        
        # Get recent chat history
        chat_history = patient_context.get("chat_history", [])
        history_text = "\n".join([f"{msg['role']}: {msg['message']}" for msg in chat_history[-5:]])
        
        # Enhanced template with patient context
        template = f"""
        You are **DiaGuide Assistant**, an expert AI clinical advisor specializing in **Type 2 Diabetes Mellitus management** based on evidence-based clinical guidelines (NICE, ADA, IDF).

        You function as a knowledgeable healthcare professional who has reviewed the patient's complete medical records. Your role is to provide expert guidance, clinical reasoning, and actionable recommendations.

        ═══════════════════════════════════════════════════════════════════

        ### PATIENT MEDICAL RECORD (Complete Information Available):
        {patient_info}

        ### CONVERSATION HISTORY:
        {history_text}

        ### CLINICAL GUIDELINES DATABASE:
        {{context}}

        ### PATIENT'S CURRENT QUESTION:
        {{query}}

        ═══════════════════════════════════════════════════════════════════

        ### YOUR ROLE & APPROACH:

        **Act as an experienced endocrinologist or diabetes specialist who:**
        - Has ALREADY reviewed all patient information above (demographics, HbA1c, medications, insulin regimen)
        - Provides direct, evidence-based answers without requesting additional information
        - Uses clinical reasoning to explain WHY recommendations are made
        - Considers the patient's complete profile when formulating responses
        - Thinks holistically about diabetes management (glycemic control, cardiovascular risk, quality of life)

        ═══════════════════════════════════════════════════════════════════

        ### MANDATORY GUIDELINES - FOLLOW STRICTLY:

        **1. NEVER ASK FOR INFORMATION YOU ALREADY HAVE:**
        - Patient demographics: ✓ Available above
        - Current medications: ✓ Available above  
        - Insulin regimen: ✓ Available above
        - HbA1c levels: ✓ Available above
        - DO NOT ask the patient to provide any of this information again

        **2. CLINICAL REASONING & DECISION-MAKING:**
        - Always explain the "WHY" behind recommendations
        - Reference specific guidelines when making recommendations
        - Consider patient's current HbA1c target and whether they're meeting it
        - Evaluate medication regimen for appropriateness and optimization opportunities
        - Think about stepwise intensification based on current control
        - Address cardiovascular and renal protective strategies

        **3. CONVERSATIONAL BEHAVIOR:**
        - Be warm, empathetic, and professional like a trusted physician
        - Use conversational language while maintaining medical accuracy
        - Acknowledge the patient's concerns and validate their experiences
        - Provide reassurance when appropriate
        - Use "you" and "your" to personalize the conversation
        - Avoid medical jargon; explain technical terms simply

        **4. RESPONSE STRUCTURE:**
        - Start by directly addressing the question (no preambles asking for info)
        - Provide immediate, actionable guidance
        - Explain clinical reasoning behind recommendations
        - Reference patient's current status (e.g., "Given your HbA1c of X%...")
        - Offer practical next steps
        - Include relevant monitoring or follow-up advice

        **5. EVIDENCE-BASED PRACTICE:**
        - Base recommendations on clinical guidelines provided in context
        - Use stepwise treatment approaches (lifestyle → oral agents → combination therapy → insulin)
        - Consider individualization based on age, comorbidities, hypoglycemia risk
        - Address both glycemic and non-glycemic outcomes
        - Mention when to consult healthcare provider for medication adjustments

        **6. COMPREHENSIVE DIABETES MANAGEMENT:**
        - Address not just medications, but lifestyle factors (diet, exercise, monitoring)
        - Consider complications screening and prevention
        - Discuss self-management education and support
        - Address mental health and quality of life aspects when relevant
        - Provide holistic care perspective

        **7. SAFETY & LIMITATIONS:**
        - Clearly state when in-person medical evaluation is needed
        - Recognize emergency situations (severe hypoglycemia, DKA symptoms)
        - Remind that this is guidance, not a substitute for their healthcare provider
        - Recommend provider consultation before major medication changes
        - Never diagnose new conditions outside of diabetes management scope

        ═══════════════════════════════════════════════════════════════════

        ### CLINICAL REASONING FRAMEWORK:

        **For each response, consider:**

        1. **Assessment**: What is the patient's current status?
        - HbA1c control (at target <7% or needs improvement?)
        - Current medication regimen (appropriate? adequate?)
        - Duration of diabetes and progression
        - Age and comorbidity considerations

        2. **Analysis**: What does the clinical evidence suggest?
        - Reference guidelines from context
        - Consider treatment algorithms and pathways
        - Evaluate current therapy against best practices
        - Identify gaps or optimization opportunities

        3. **Recommendations**: What specific actions should be taken?
        - Provide clear, prioritized recommendations
        - Explain rationale using clinical reasoning
        - Include both immediate and long-term strategies
        - Personalize to patient's specific situation

        4. **Monitoring**: What should be tracked?
        - Relevant biomarkers or symptoms
        - Timing of follow-up
        - Warning signs to watch for

        ═══════════════════════════════════════════════════════════════════

        ### RESPONSE TONE & STYLE:

        **DO:**
        - ✓ Be conversational and approachable
        - ✓ Show empathy and understanding
        - ✓ Explain complex concepts simply
        - ✓ Provide specific, actionable advice
        - ✓ Use clinical reasoning transparently
        - ✓ Reference patient's data naturally ("Given your HbA1c of 7%...")
        - ✓ Offer encouragement and positive reinforcement
        - ✓ Structure information clearly with organized points

        **DON'T:**
        - ✗ Ask for information already provided in patient record
        - ✗ Use overly technical language without explanation
        - ✗ Provide vague or generic advice
        - ✗ Sound robotic or impersonal
        - ✗ Make recommendations beyond diabetes management scope
        - ✗ Create anxiety unnecessarily
        - ✗ Contradict established guidelines without explanation

        ═══════════════════════════════════════════════════════════════════

        ### EXAMPLE RESPONSE PATTERN:

        **Good Response:**
        "Looking at your current profile, your HbA1c of 7% is right at the target, which is excellent! [Clinical Assessment]

        Given that you're on Metformin 500mg twice daily, you're following the recommended first-line treatment approach for Type 2 diabetes. [Guideline Reference]

        For your question about [topic], here's what the clinical evidence suggests... [Clinical Reasoning]

        I'd recommend... [Specific Action Items]

        Keep monitoring... [Follow-up Guidance]"

        **Bad Response:**
        "Hello! To provide you with personalized advice, I need to know:
        - What medications are you taking?
        - What's your current HbA1c?
        ..."
        [This is WRONG - You already have this information!]

        ═══════════════════════════════════════════════════════════════════

        Now, based on the patient's complete medical record above, the clinical guidelines, and conversation history, provide your expert clinical guidance for their current question. 

        Remember: You are a knowledgeable diabetes specialist who has all the patient information. Provide direct, evidence-based, clinically reasoned advice with warmth and professionalism.

        ### Your Clinical Response:
        """
                
        formatted_prompt = template.format(context=guideline_context, query=query)
        response = model.generate_content(formatted_prompt)
        
        if not response or not getattr(response, "text", None):
            return "⚠️ Sorry, I couldn't generate a valid response. Please try again."
        
        return response.text
        
    except Exception as e:
        logging.error(f"Response generation failed: {e}")
        return f"⚠️ An internal error occurred: {e}"

# =========================================================
# Session State Initialization
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "page" not in st.session_state:
    st.session_state.page = "login"
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# Page Functions
# =========================================================

def login_page():
    """Login and Registration Page"""
    st.title("💉 DiaGuide Assistant")
    st.subheader("AI-powered Type 2 Diabetes Management")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.header("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary"):
            if not email or not password:
                st.error("Please enter both email and password")
            else:
                user = get_user_by_email(email)
                if user and user["password_hash"] == hash_password(password):
                    st.session_state.logged_in = True
                    st.session_state.user_id = user["user_id"]
                    st.session_state.user_email = user["email"]
                    
                    # Check if profile is completed
                    if is_profile_completed(user["user_id"]):
                        st.session_state.page = "chat"
                    else:
                        st.session_state.page = "profile"
                    st.rerun()
                else:
                    st.error("Invalid email or password")
    
    with tab2:
        st.header("Register")
        new_email = st.text_input("Email", key="register_email")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Register", type="primary"):
            if not new_email or not new_password:
                st.error("Please fill all fields")
            elif not validate_email(new_email):
                st.error("Please enter a valid email address")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                existing_user = get_user_by_email(new_email)
                if existing_user:
                    st.error("Email already registered")
                else:
                    user_id = create_user(new_email, hash_password(new_password))
                    st.success("✅ Registration successful! Please login.")

def profile_setup_page():
    """Profile Setup Page"""
    st.title("👤 Complete Your Profile")
    st.info("Please complete your profile to start using DiaGuide Assistant")
    
    # Initialize session state for insulin and medications
    if "insulin_entries" not in st.session_state:
        # Load existing data if profile exists
        existing_insulin = get_insulin_schedule(st.session_state.user_id)
        st.session_state.insulin_entries = existing_insulin if existing_insulin else []
    
    if "medication_entries" not in st.session_state:
        existing_meds = get_medications(st.session_state.user_id)
        st.session_state.medication_entries = existing_meds if existing_meds else []
    
    # Load existing profile data
    existing_profile = get_user_profile(st.session_state.user_id)
    
    with st.form("profile_form"):
        st.subheader("📋 Demographic Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, 
                                 value=existing_profile.get("age", 30) if existing_profile else 30)
            weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, 
                                    value=float(existing_profile.get("weight", 70.0)) if existing_profile else 70.0)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                                 index=["Male", "Female", "Other"].index(existing_profile.get("gender", "Male")) if existing_profile and existing_profile.get("gender") in ["Male", "Female", "Other"] else 0)
            height = st.number_input("Height (cm)", min_value=1.0, max_value=300.0, 
                                    value=float(existing_profile.get("height", 170.0)) if existing_profile else 170.0)
        
        st.subheader("🩺 Diabetes Information")
        col3, col4 = st.columns(2)
        with col3:
            diabetes_type = st.selectbox("Diabetes Type", ["Type 2", "Type 1", "Gestational", "Other"],
                                        index=["Type 2", "Type 1", "Gestational", "Other"].index(existing_profile.get("diabetes_type", "Type 2")) if existing_profile and existing_profile.get("diabetes_type") in ["Type 2", "Type 1", "Gestational", "Other"] else 0)
            diagnosis_year = st.number_input("Year of Diagnosis", min_value=1950, max_value=datetime.now().year, 
                                            value=existing_profile.get("diagnosis_year", 2020) if existing_profile else 2020)
        with col4:
            hba1c = st.number_input("Latest HbA1c (%)", min_value=0.0, max_value=20.0, 
                                   value=float(existing_profile.get("hba1c", 7.0)) if existing_profile else 7.0, step=0.1)
        
        submitted = st.form_submit_button("💾 Save Profile", type="primary", use_container_width=True)
        
        if submitted:
            # Save profile
            profile_data = {
                "age": age,
                "gender": gender,
                "weight": weight,
                "height": height,
                "diabetes_type": diabetes_type,
                "diagnosis_year": diagnosis_year,
                "hba1c": hba1c
            }
            save_user_profile(st.session_state.user_id, profile_data)
            
            # Clear existing insulin and medications
            delete_insulin_by_user(st.session_state.user_id)
            delete_medications_by_user(st.session_state.user_id)
            
            # Save insulin schedule
            for ins in st.session_state.insulin_entries:
                add_insulin(st.session_state.user_id, ins["insulin_name"], ins["dosage"], ins["timing"])
            
            # Save medications
            for med in st.session_state.medication_entries:
                add_medication(st.session_state.user_id, med["medicine_name"], med["dosage"], med["frequency"])
            
            st.success("✅ Profile saved successfully!")
            st.session_state.page = "chat"
            st.rerun()
    
    # Insulin Schedule Section (Outside form for dynamic updates)
    st.subheader("💉 Insulin Schedule")
    
    # Display existing insulin entries
    if st.session_state.insulin_entries:
        for idx, ins in enumerate(st.session_state.insulin_entries):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.text_input("Insulin Name", value=ins.get("insulin_name", ""), key=f"display_ins_name_{idx}", disabled=True)
            with col2:
                st.text_input("Dosage", value=ins.get("dosage", ""), key=f"display_ins_dose_{idx}", disabled=True)
            with col3:
                st.text_input("Timing", value=ins.get("timing", ""), key=f"display_ins_time_{idx}", disabled=True)
            with col4:
                if st.button("🗑️", key=f"del_ins_{idx}"):
                    st.session_state.insulin_entries.pop(idx)
                    st.rerun()
    else:
        st.info("No insulin added yet. Add your insulin regimen below.")
    
    # Add new insulin
    with st.expander("➕ Add Insulin", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_ins_name = st.text_input("Insulin Name", key="new_ins_name", placeholder="e.g., Lantus, Humalog")
        with col2:
            new_ins_dosage = st.text_input("Dosage", key="new_ins_dosage", placeholder="e.g., 10 units")
        with col3:
            new_ins_timing = st.text_input("Timing", key="new_ins_timing", placeholder="e.g., Before breakfast")
        
        if st.button("Add Insulin", type="secondary"):
            if new_ins_name and new_ins_dosage and new_ins_timing:
                st.session_state.insulin_entries.append({
                    "insulin_name": new_ins_name,
                    "dosage": new_ins_dosage,
                    "timing": new_ins_timing
                })
                st.success(f"✅ Added {new_ins_name}")
                st.rerun()
            else:
                st.error("Please fill in all insulin fields")
    
    st.divider()
    
    # Medications Section (Outside form for dynamic updates)
    st.subheader("💊 Medications")
    
    # Display existing medication entries
    if st.session_state.medication_entries:
        for idx, med in enumerate(st.session_state.medication_entries):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.text_input("Medicine Name", value=med.get("medicine_name", ""), key=f"display_med_name_{idx}", disabled=True)
            with col2:
                st.text_input("Dosage", value=med.get("dosage", ""), key=f"display_med_dose_{idx}", disabled=True)
            with col3:
                st.text_input("Frequency", value=med.get("frequency", ""), key=f"display_med_freq_{idx}", disabled=True)
            with col4:
                if st.button("🗑️", key=f"del_med_{idx}"):
                    st.session_state.medication_entries.pop(idx)
                    st.rerun()
    else:
        st.info("No medications added yet. Add your medications below.")
    
    # Add new medication
    with st.expander("➕ Add Medication", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_med_name = st.text_input("Medicine Name", key="new_med_name", placeholder="e.g., Metformin")
        with col2:
            new_med_dosage = st.text_input("Dosage", key="new_med_dosage", placeholder="e.g., 500mg")
        with col3:
            new_med_freq = st.text_input("Frequency", key="new_med_frequency", placeholder="e.g., Twice daily")
        
        if st.button("Add Medication", type="secondary"):
            if new_med_name and new_med_dosage and new_med_freq:
                st.session_state.medication_entries.append({
                    "medicine_name": new_med_name,
                    "dosage": new_med_dosage,
                    "frequency": new_med_freq
                })
                st.success(f"✅ Added {new_med_name}")
                st.rerun()
            else:
                st.error("Please fill in all medication fields")

def chat_page():
    """Main Chat Interface"""
    st.title("💉 DiaGuide Assistant")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.write(f"**User:** {st.session_state.user_email}")
        
        if st.button("View/Edit Profile"):
            st.session_state.page = "profile"
            st.rerun()
        
        if st.button("Clear Chat History"):
            clear_chat_history(st.session_state.user_id)
            st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.user_email = None
            st.session_state.page = "login"
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Display patient info summary
        profile = get_user_profile(st.session_state.user_id)
        if profile:
            st.subheader("📊 Your Profile Summary")
            st.write(f"**Age:** {profile.get('age')} years")
            st.write(f"**Diabetes Type:** {profile.get('diabetes_type')}")
            st.write(f"**HbA1c:** {profile.get('hba1c')}%")
    
    # Load chat history from database
    if not st.session_state.messages:
        history = get_chat_history(st.session_state.user_id, limit=20)
        for msg in history:
            st.session_state.messages.append({
                "role": msg["role"],
                "content": msg["message"]
            })
    
    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"{msg['content']}_{msg['role']}")
        else:
            message(msg["content"], is_user=False, key=f"{msg['content']}_{msg['role']}")
    
    # Chat input
    query = st.chat_input("Ask about Type 2 Diabetes management...")
    
    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        save_chat_message(st.session_state.user_id, "user", query)
        
        # Generate response
        with st.spinner("Analyzing guidelines and your profile..."):
            response = generate_response(query, st.session_state.user_id)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_chat_message(st.session_state.user_id, "assistant", response)
        
        st.rerun()

# =========================================================
# Main App Router
# =========================================================

def main():
    st.set_page_config(
        page_title="DiaGuide Assistant",
        page_icon="💉",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if not st.session_state.logged_in:
        login_page()
    elif st.session_state.page == "profile":
        profile_setup_page()
    elif st.session_state.page == "chat":
        chat_page()

if __name__ == "__main__":
    main()