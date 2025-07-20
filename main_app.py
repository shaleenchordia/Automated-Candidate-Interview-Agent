import streamlit as st
import sys
import os
from bson import ObjectId

# Add the backend directory to Python path to fix imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Now try importing the modules
try:
    from app.services.dynamic_question_generator import DynamicQuestionGenerator
    from app.services.interview_service import InterviewService
    from app.services.resume_analyzer import ResumeAnalyzer
    from app.services.candidate_service import CandidateService
    from app.services.speech_service import SpeechService
    from app.utils.resume_parser import parse_resume
    from app.ml.inference.skills_detector import SkillDetector
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure all dependencies are installed and the file structure is correct.")
    st.stop()

# Initialize services
dynamic_question_generator = DynamicQuestionGenerator()
interview_service = InterviewService()
resume_analyzer = ResumeAnalyzer()
candidate_service = CandidateService()
speech_service = SpeechService()
skills_detector = SkillDetector()  # New comprehensive skill detector

def main():
    st.title("AI Interview Assistant")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Upload Resume", "Interview", "Results"])
    
    if page == "Upload Resume":
        resume_page()
    elif page == "Interview":
        interview_page()
    else:
        results_page()

def resume_page():
    st.header("Upload Resume & Job Description")
    
    # Candidate Name Input
    candidate_name = st.text_input("Enter Candidate's Full Name", key="candidate_name")
    
    # Resume upload
    resume_file = st.file_uploader("Upload Resume (PDF/DOC)", type=["pdf", "doc", "docx", "txt"])
    if resume_file:
        try:
            resume_text = parse_resume(resume_file)
            st.session_state['resume_text'] = resume_text
            
            # Analyze resume with comprehensive skill detection
            with st.spinner("🔍 Analyzing resume..."):
                # Use our new comprehensive skill detector
                skill_analysis = skills_detector.analyze_resume(resume_text)
                st.session_state['skill_analysis'] = skill_analysis
                
                # Keep old analysis for compatibility
                old_analysis = resume_analyzer.analyze(resume_text)
                st.session_state['resume_analysis'] = old_analysis
                
                st.success("✅ Resume analyzed successfully!")
                
                # Display comprehensive skill analysis
                st.subheader("🎯 Skill Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Skills Detected", skill_analysis['total_categories_detected'])
                with col2:
                    st.metric("Technical Level", skill_analysis['overall_technical_level'])
                with col3:
                    st.metric("Top Skills", min(10, len(skill_analysis['top_skills'])))
                with col4:
                    confidence_avg = sum(data['confidence'] for data in skill_analysis['skill_scores'].values()) / len(skill_analysis['skill_scores']) if skill_analysis['skill_scores'] else 0
                    st.metric("Avg Confidence", f"{confidence_avg:.2f}")
                
                # Summary
                st.info(f"📋 **Summary:** {skill_analysis['summary']}")
                
                # Top skills display
                if skill_analysis['top_skills']:
                    st.subheader("Skills")
                    top_skills_df_data = []
                    for i, skill in enumerate(skill_analysis['top_skills'][:10], 1):
                        skill_data = skill_analysis['skill_scores'][skill]
                        top_skills_df_data.append({
                            "Rank": i,
                            "Skill Category": skill,
                            "Confidence": f"{skill_data['confidence']:.3f}",
                            "Experience Level": skill_data['experience_level'],
                            "Evidence Count": skill_data['keyword_count'],
                            "Subcategories": ", ".join(skill_data['subcategories'][:3])  # Show first 3
                        })
                    
                    import pandas as pd
                    st.dataframe(pd.DataFrame(top_skills_df_data), use_container_width=True)
                
                # Detailed breakdown by domain
                st.subheader("📊 Skills by Domain")
                
                domains = {
                    'Programming Languages': [
                        'Python Programming', 'JavaScript Programming', 'Java Programming', 
                        'C/C++ Programming', 'C# Programming', 'Go Programming', 'Rust Programming',
                        'PHP Programming', 'Ruby Programming', 'Swift Programming', 'Kotlin Programming',
                        'Scala Programming', 'R Programming'
                    ],
                    'AI/ML & Data Science': [
                        'Machine Learning', 'Deep Learning', 'Data Science', 'Computer Vision', 
                        'Natural Language Processing', 'Reinforcement Learning', 'Generative AI',
                        'Big Data Technologies', 'Data Engineering', 'Data Analytics & BI'
                    ],
                    'Cloud & Infrastructure': [
                        'Amazon Web Services', 'Microsoft Azure', 'Google Cloud Platform',
                        'Multi-Cloud & Hybrid', 'DevOps & CI/CD', 'Containerization & Orchestration',
                        'Infrastructure as Code', 'Monitoring & Observability'
                    ],
                    'Web & Mobile Development': [
                        'Frontend Development', 'React Ecosystem', 'Vue.js Ecosystem', 'Angular Ecosystem',
                        'UI/UX Design', 'Mobile Development', 'Backend Development'
                    ]
                }
                
                for domain_name, categories in domains.items():
                    found_skills = []
                    for category in categories:
                        if category in skill_analysis['skill_scores']:
                            skill_data = skill_analysis['skill_scores'][category]
                            found_skills.append((category, skill_data['confidence'], skill_data['experience_level']))
                    
                    if found_skills:
                        with st.expander(f"🔹 {domain_name} ({len(found_skills)} skills detected)"):
                            for skill, confidence, level in sorted(found_skills, key=lambda x: x[1], reverse=True):
                                evidence = skill_analysis['skill_scores'][skill]['evidence'][:5]
                                st.write(f"**{skill}** - Confidence: {confidence:.3f} | Level: {level}")
                                st.caption(f"Evidence: {', '.join(evidence)}")
                
        except Exception as e:
            st.error(f"❌ Error processing resume: {str(e)}")
            st.error("Please check the file format and try again.")
    
    # Job Category Selection
    st.subheader("Select Job Details")

    job_categories = [
        "Software Development",
        "Data & Analytics",
        "Infrastructure & DevOps",
        "Quality Assurance & Testing",
        "Product & Project Management",
        "Design & UX"
    ]

    selected_category = st.selectbox(
        "1. Select Job Category:",
        options=job_categories
    )

    # Experience Level
    experience_levels = [
        "Entry Level (0-2 years)",
        "Junior (2-4 years)",
        "Mid-Level (4-6 years)",
        "Senior (6-10 years)",
        "Lead/Principal (10+ years)"
    ]
    
    selected_experience = st.selectbox(
        "3. Experience:",
        options=experience_levels
    )

    # Skills Selection based on role
    st.subheader("Technical Requirements")
    
    # Define skills based on categories
    skill_options = {
        "Software Development": {
            "languages": ["Java", "Python", "JavaScript", "TypeScript", "C#", "Go", "Ruby", "PHP"],
            "frameworks": ["React", "Angular", "Vue.js", "Django", "Spring", "Node.js", ".NET"],
            "databases": ["MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle"],
            "tools": ["Git", "Docker", "Kubernetes", "Jenkins", "JIRA"]
        },
        "Data & Analytics": {
            "languages": ["Python", "R", "SQL", "Scala"],
            "frameworks": ["TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "Spark"],
            "tools": ["Jupyter", "Tableau", "Power BI", "Airflow"],
            "concepts": ["Machine Learning", "Deep Learning", "Statistics", "Data Visualization"]
        },
        "Infrastructure & DevOps": {
            "languages": ["Python", "Shell Script", "PowerShell", "Go"],
            "frameworks": ["Terraform", "Ansible", "CloudFormation", "Puppet", "Chef"],
            "tools": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "GitLab CI", "Prometheus", "Grafana"]
        },
        "Quality Assurance & Testing": {
            "languages": ["Python", "Java", "JavaScript"],
            "frameworks": ["Selenium", "Cypress", "JUnit", "PyTest", "TestNG", "Robot Framework"],
            "tools": ["JIRA", "TestRail", "Postman", "JMeter", "SonarQube"]
        },
        "Product & Project Management": {
            "frameworks": ["Agile", "Scrum", "Kanban", "Waterfall", "Lean"],
            "tools": ["JIRA", "Confluence", "Trello", "Microsoft Project", "Asana", "Monday.com"],
            "concepts": ["Risk Management", "Stakeholder Management", "Sprint Planning", "Resource Allocation"]
        },
        "Design & UX": {
            "frameworks": ["Design Thinking", "User-Centered Design", "Atomic Design"],
            "tools": ["Figma", "Sketch", "Adobe XD", "InVision", "Zeplin", "Adobe Creative Suite"],
            "concepts": ["Wireframing", "Prototyping", "User Research", "Usability Testing", "Information Architecture"]
        }
    }

    col1, col2 = st.columns(2)
    
    with col1:
        # Combine all relevant skill types for the selected category
        all_skills = []
        category_skills = skill_options.get(selected_category, {})
        if "languages" in category_skills:
            all_skills.extend(category_skills["languages"])
        if "frameworks" in category_skills:
            all_skills.extend(category_skills["frameworks"])
        if "concepts" in category_skills:
            all_skills.extend(category_skills["concepts"])
            
        selected_skills = st.multiselect(
            "4. Required Technical Skills:",
            options=all_skills
        )

    with col2:
        selected_tools = st.multiselect(
            "5. Required Tools:",
            options=skill_options.get(selected_category, {}).get("tools", [])
        )

    work_models = [
        "On-site",
        "Remote",
        "Hybrid"
    ]

    selected_work_model = st.selectbox(
        "4. Work Model:",
        options=work_models
    )

    # Build job description from selections
    if selected_category:
        if not st.session_state.get("candidate_name"):
            st.error("Please enter candidate's name before proceeding.")
            return
            
        # Create job profile data with top 2 skills from comprehensive analysis
        top_skills_for_storage = []
        if 'skill_analysis' in st.session_state:
            # Extract top 2 skills for MongoDB storage
            skill_analysis = st.session_state['skill_analysis']
            top_skills_for_storage = skill_analysis['top_skills'][:2]  # Only top 2
            st.info(f"📊 Top 2 skills for storage: {', '.join(top_skills_for_storage)}")
        
        job_profile = {
            "candidate_name": st.session_state.get("candidate_name"),
            "category": selected_category,
            "experience_level": selected_experience,
            "work_model": selected_work_model,
            "technical_skills": selected_skills,
            "required_tools": selected_tools,
            "resume_uploaded": True if "resume_text" in st.session_state else False,
            "skills_identified": top_skills_for_storage if top_skills_for_storage else st.session_state.get("resume_analysis", {}).get("skills", [])
        }
        
        job_description = f"""Candidate: {job_profile['candidate_name']}
Category: {job_profile['category']}
Experience: {job_profile['experience_level']}
Work Model: {job_profile['work_model']}

Technical Skills Required:
{', '.join(job_profile['technical_skills'])}

Required Tools:
{', '.join(job_profile['required_tools'])}

Skills Identified from Resume:
{', '.join(job_profile['skills_identified']) if job_profile['resume_uploaded'] else 'Resume not uploaded'}
"""
        st.session_state['job_description'] = job_description
        st.session_state['job_profile'] = job_profile
        
        # Show preview and generate questions
        st.subheader("Job Profile Preview")
        st.info(job_description)

        if st.button("Start Interview"):
            if not selected_skills:
                st.error("Please select at least one technical skill.")
                return
                
            # Store candidate profile in MongoDB
            try:
                # Create enhanced candidate profile with skill analysis
                profile_data = {
                    "name": job_profile['candidate_name'],
                    "job_profile": job_profile,
                    "status": "profile_created",
                    "interview_completed": False,
                    "quiz_score": 0,
                    "total_questions": 3  # Fixed number of questions
                }
                
                # Add essential skill summary for MongoDB storage
                if 'skill_analysis' in st.session_state:
                    skill_analysis = st.session_state['skill_analysis']
                    
                    # Store only essential skill information in MongoDB
                    profile_data["skill_summary"] = {
                        "top_2_skills": skill_analysis['top_skills'][:2],  # Only top 2 skills
                        "technical_level": skill_analysis['overall_technical_level'],
                        "total_categories": skill_analysis['total_categories_detected'],
                        "job_category_preference": job_profile['category']  # User's job preference
                    }
                else:
                    # If no skill analysis, still store job preference
                    profile_data["skill_summary"] = {
                        "top_2_skills": [],
                        "technical_level": "Unknown",
                        "total_categories": 0,
                        "job_category_preference": job_profile['category']
                    }
                
                candidate_id = candidate_service.create_candidate_profile(
                    name=profile_data["name"],
                    job_profile=profile_data
                )
                st.session_state['candidate_id'] = str(candidate_id)
                
                # Enhanced question generation using dynamic LLM generator
                with st.spinner("Generating personalized interview questions..."):
                    # Extract parameters for dynamic question generation
                    job_category = job_profile['category']
                    experience_level = job_profile['experience_level']
                    
                    # Get candidate skills from comprehensive analysis
                    candidate_skills = []
                    technical_level = "Mid-level"  # Default
                    
                    if 'skill_analysis' in st.session_state:
                        skill_analysis = st.session_state['skill_analysis']
                        candidate_skills = skill_analysis['top_skills'][:5]  # Top 5 skills for question generation
                        technical_level = skill_analysis['overall_technical_level']
                    else:
                        # Fallback to basic technical skills if no analysis available
                        candidate_skills = job_profile.get('technical_skills', [])
                    
                    # Generate personalized questions using new dynamic generator
                    questions = dynamic_question_generator.generate_personalized_questions(
                        job_category=job_category,
                        candidate_skills=candidate_skills,
                        experience_level=experience_level,
                        technical_level=technical_level,
                        num_questions=3
                    )
                    
                    # Ensure exactly 3 questions
                    if len(questions) >= 3:
                        st.session_state['questions'] = questions[:3]  # Take first 3 questions
                        st.success("✨ Interview questions generated successfully!")
                    else:
                        st.error("Failed to generate enough questions. Please try again.")
                        return
            except Exception as e:
                st.error(f"""Error setting up interview: {str(e)}
Please ensure:
1. You've selected an appropriate role
2. The job description is detailed enough
3. You've included required skills and experience""")
                st.info("Try using the template provided and fill in all the sections.")

def interview_page():
    st.header("Interview Session")
    
    if 'questions' not in st.session_state:
        st.warning("Please upload a resume and job description first!")
        return
    
    if 'candidate_id' not in st.session_state:
        st.warning("Candidate profile not found. Please start from the beginning.")
        return
    
    if 'current_question' not in st.session_state:
        st.session_state['current_question'] = 0
        st.session_state['responses'] = []
    
    questions = st.session_state['questions'][:3]  # Ensure only 3 questions are used
    
    if st.session_state['current_question'] < 3:  # Fixed number of questions
        current_q = questions[st.session_state['current_question']]
        st.subheader(f"Question {st.session_state['current_question'] + 1} of 3:")
        st.write(current_q)
        
        # Text response
        response = st.text_area("Your answer:", key=f"q_{st.session_state['current_question']}")
        
        # Show previously transcribed text if available
        transcribed_key = f"transcribed_{st.session_state['current_question']}"
        if transcribed_key in st.session_state:
            st.info(f"📝 **Previously transcribed text:** {st.session_state[transcribed_key]}")
            st.caption("You can copy this text and paste it into the answer box above.")
        
        # Voice input option
        st.write("---")
        st.write("🎤 **Voice Input Option**")
        
        # Check if speech services are available
        if not speech_service.is_available():
            st.warning("⚠️ Voice input requires OpenAI API key or Hugging Face API key.")
            st.info("Add OPENAI_API_KEY to your .env file for the best speech-to-text experience!")
        else:
            st.info("🎙️ Upload an audio file with your spoken answer")
            
            # File uploader for audio
            uploaded_audio = st.file_uploader(
                "Upload audio file (WAV, MP3, M4A, OGG)", 
                type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
                key=f"audio_upload_{st.session_state['current_question']}",
                help="Record your answer using your phone's voice recorder or computer microphone, then upload the file here."
            )
            
            if uploaded_audio is not None:
                st.audio(uploaded_audio)
                
                if st.button("🔊 Transcribe Audio with AI", key=f"transcribe_file_{st.session_state['current_question']}"):
                    with st.spinner("Converting speech to text using AI..."):
                        try:
                            # Read audio file bytes
                            audio_bytes = uploaded_audio.read()
                            
                            # Use real speech-to-text service
                            transcribed_text = speech_service.transcribe_audio(audio_bytes)
                            
                            if transcribed_text and transcribed_text.strip() and not transcribed_text.startswith("Mock transcription"):
                                # Check if it's an error message
                                if any(word in transcribed_text.lower() for word in ['error', 'failed', 'api error']):
                                    st.error(f"❌ {transcribed_text}")
                                    st.info("💡 Try uploading a clearer audio file or check your API keys.")
                                else:
                                    st.success("✅ Audio transcribed successfully!")
                                    st.write("**Transcribed text:**")
                                    st.write(f'"{transcribed_text}"')
                                    
                                    # Store transcribed text in session state for later use
                                    st.session_state[f"transcribed_{st.session_state['current_question']}"] = transcribed_text
                                    st.info("💡 Copy the transcribed text above and paste it into the answer box below.")
                                
                            else:
                                if transcribed_text.startswith("Mock transcription"):
                                    st.warning("⚠️ Using mock transcription - please add your API keys for real transcription.")
                                else:
                                    st.error("❌ No speech detected in the audio. Please try again with a clearer recording.")
                                
                        except Exception as e:
                            st.error(f"❌ Error in speech recognition: {str(e)}")
                            st.error("Please try again or use text input instead.")
            
            # Additional instructions
            with st.expander("📝 How to use Real Speech-to-Text"):
                st.write("""
                **Real AI Speech-to-Text Feature**
                
                **How to get started:**
                1. **Get OpenAI API Key** (Recommended):
                   - Go to https://platform.openai.com/api-keys
                   - Create an account and get your API key
                   - Add it to your .env file: `OPENAI_API_KEY=your_key_here`
                   - Uses OpenAI Whisper - the best speech recognition available
                
                2. **Alternative - Use Hugging Face** (Current):
                   - You already have a Hugging Face API key configured
                   - Will use Whisper via Hugging Face (may be slower)
                
                **How it works:**
                1. Record your answer on any device
                2. Upload the audio file
                3. Click "Transcribe Audio with AI"
                4. Get real AI transcription of your speech
                5. Copy and paste into the answer box
                
                **Supported formats:**
                - WAV (best quality)
                - MP3, M4A, OGG, FLAC
                - Any audio recording from phone or computer
                
                **Tips for best results:**
                - Speak clearly and at normal pace
                - Minimize background noise
                - Keep recordings under 25MB
                - Use WAV format for best accuracy
                
                **About the AI Models:**
                - **OpenAI Whisper**: State-of-the-art accuracy, multiple languages
                - **Hugging Face Whisper**: Same model, free tier available
                - Both handle accents, background noise, and various audio quality
                """)
        
        if st.button("Submit Answer"):
            if response:
                with st.spinner("Evaluating response..."):
                    # Get detailed evaluation based on response and question context
                    evaluation = interview_service.evaluate_response(response, current_q)
                    
                    # Validate evaluation scores
                    if not evaluation or 'scores' not in evaluation:
                        st.error("Error in evaluation. Please try again.")
                        return
                        
                    st.session_state['responses'].append({
                        'question': current_q,
                        'response': response,
                        'evaluation': evaluation
                    })
                    st.session_state['current_question'] += 1
                    st.experimental_rerun()
            else:
                st.warning("Please provide an answer before proceeding.")
    else:
        st.success("Interview completed! Check the Results page for feedback.")

def results_page():
    st.header("Interview Results & Feedback")
    
    if 'responses' not in st.session_state or not st.session_state['responses']:
        st.warning("No interview data available. Please complete the interview first!")
        return

    if 'candidate_id' not in st.session_state:
        st.warning("Candidate profile not found. Please start from the beginning.")
        return
    
    # Overall Score
    total_score = 0
    num_responses = len(st.session_state['responses'])
    
    for idx, response_data in enumerate(st.session_state['responses']):
        st.subheader(f"Question {idx + 1}:")
        st.write(response_data['question'])
        
        st.write("Your Response:")
        st.write(response_data['response'])
        
        evaluation = response_data['evaluation']
        scores = evaluation['scores']
        feedback = evaluation.get('feedback', {})
        
        # Create two columns: one for scores and one for feedback
        score_col, feedback_col = st.columns([1, 2])
        
        with score_col:
            st.write("Scores:")
            st.metric("Relevance", f"{scores['relevance']}/10")
            st.metric("Clarity", f"{scores['clarity']}/10")
            st.metric("Technical", f"{scores['technical']}/10")
            st.metric("Problem Solving", f"{scores['problem_solving']}/10")
        
        with feedback_col:
            st.write("Detailed Feedback:")
            if feedback:
                st.write("🎯 Relevance:", feedback.get('relevance', 'No feedback available'))
                st.write("📝 Clarity:", feedback.get('clarity', 'No feedback available'))
                st.write("💡 Technical:", feedback.get('technical', 'No feedback available'))
                st.write("🔍 Problem Solving:", feedback.get('problem_solving', 'No feedback available'))
        
        # Overall feedback for this question
        if 'overall_feedback' in evaluation:
            st.write("Overall Feedback:")
            st.info(evaluation['overall_feedback'])
        
        total_score += evaluation['overall_score']
        st.markdown("---")
    
    # Final Score and Summary
    st.header("Final Assessment")
    
    # Create three columns for the final metrics
    score_col, strength_col, improve_col = st.columns(3)
    
    with score_col:
        average_score = total_score / num_responses
        st.metric("Overall Score", f"{average_score:.1f}/10")
        
        # Add score interpretation
        if average_score >= 8:
            st.success("Excellent Performance!")
        elif average_score >= 6:
            st.info("Good Performance")
        elif average_score >= 4:
            st.warning("Needs Improvement")
        else:
            st.error("Significant Improvement Required")
    
    with strength_col:
        st.subheader("Key Strengths")
        # Calculate highest scoring areas
        avg_scores = {
            "Relevance": sum(r['evaluation']['scores']['relevance'] for r in st.session_state['responses']) / num_responses,
            "Clarity": sum(r['evaluation']['scores']['clarity'] for r in st.session_state['responses']) / num_responses,
            "Technical": sum(r['evaluation']['scores']['technical'] for r in st.session_state['responses']) / num_responses,
            "Problem Solving": sum(r['evaluation']['scores']['problem_solving'] for r in st.session_state['responses']) / num_responses
        }
        strengths = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        for area, score in strengths:
            st.write(f"✅ {area}: {score:.1f}/10")
    
    with improve_col:
        st.subheader("Areas to Improve")
        improvements = sorted(avg_scores.items(), key=lambda x: x[1])[:2]
        for area, score in improvements:
            st.write(f"📈 {area}: {score:.1f}/10")
    
    # Enhanced Skills Analysis Section
    if 'skill_analysis' in st.session_state:
        st.header("🎯 Comprehensive Skills Assessment")
        skill_analysis = st.session_state['skill_analysis']
        
        # Skills overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Skills Detected", skill_analysis['total_categories_detected'])
        with col2:
            st.metric("Technical Level", skill_analysis['overall_technical_level'])
        with col3:
            st.metric("Interview Score", f"{average_score:.1f}/10")
        with col4:
            # Calculate skill-performance alignment
            high_confidence_skills = sum(1 for data in skill_analysis['skill_scores'].values() if data['confidence'] >= 0.7)
            alignment = min(100, (high_confidence_skills / max(1, skill_analysis['total_categories_detected'])) * 100)
            st.metric("Skill Alignment", f"{alignment:.0f}%")
        
        # Top skills performance correlation
        st.subheader("🏆 Top Skills vs Interview Performance")
        top_skills_data = []
        for i, skill in enumerate(skill_analysis['top_skills'][:8], 1):
            skill_data = skill_analysis['skill_scores'][skill]
            # Simple correlation: higher confidence skills should correlate with better performance
            expected_performance = min(10, skill_data['confidence'] * 10)
            performance_gap = average_score - expected_performance
            
            top_skills_data.append({
                "Rank": i,
                "Skill": skill,
                "Confidence": f"{skill_data['confidence']:.3f}",
                "Level": skill_data['experience_level'],
                "Expected Score": f"{expected_performance:.1f}",
                "Actual Score": f"{average_score:.1f}",
                "Gap": f"{performance_gap:+.1f}"
            })
        
        import pandas as pd
        st.dataframe(pd.DataFrame(top_skills_data), use_container_width=True)
        
        # Recommendations based on skill analysis
        st.subheader("📋 Personalized Recommendations")
        
        if average_score >= 7 and skill_analysis['overall_technical_level'] in ['Senior', 'Mid-level']:
            st.success("🎉 **Strong Performance!** Your interview performance aligns well with your detected skill level.")
            st.info("💡 **Next Steps:** Consider applying for senior roles or leadership positions in your strong skill areas.")
        elif average_score < 6 and skill_analysis['overall_technical_level'] == 'Senior':
            st.warning("📊 **Performance Gap Detected:** Your resume shows senior-level skills, but interview performance suggests room for improvement.")
            st.info("💡 **Recommendations:** Practice articulating your technical experience and prepare specific examples for each skill area.")
        elif average_score >= 6 and skill_analysis['overall_technical_level'] == 'Junior':
            st.success("📈 **Exceeding Expectations!** Your interview performance exceeds what your resume skill level suggests.")
            st.info("💡 **Opportunity:** Consider highlighting more advanced projects on your resume to match your interview performance.")
        
        # Skills to focus on
        low_confidence_skills = [skill for skill, data in skill_analysis['skill_scores'].items() 
                               if data['confidence'] < 0.5 and skill in skill_analysis['top_skills'][:10]]
        
        if low_confidence_skills:
            st.subheader("🎯 Skills to Strengthen")
            st.info("Based on your resume analysis, consider strengthening these areas:")
            for skill in low_confidence_skills[:5]:
                st.write(f"• {skill}")
    
    # Store final results in MongoDB
    try:
        # Enhanced results with essential skill analysis for MongoDB
        results_data = {
            "quiz_score": average_score,
            "total_questions": num_responses,
            "detailed_scores": avg_scores,
            "responses": st.session_state['responses']
        }
        
        # Add only essential skill information to MongoDB (not full analysis)
        if 'skill_analysis' in st.session_state:
            skill_analysis = st.session_state['skill_analysis']
            job_preference = st.session_state.get('job_profile', {}).get('category', 'Unknown')
            
            results_data["skill_summary"] = {
                "top_2_skills": skill_analysis['top_skills'][:2],  # Only top 2 for storage
                "technical_level": skill_analysis['overall_technical_level'],
                "interview_score": average_score,
                "job_category_preference": job_preference  # User's job preference
            }
        else:
            # Fallback if no skill analysis
            job_preference = st.session_state.get('job_profile', {}).get('category', 'Unknown')
            results_data["skill_summary"] = {
                "top_2_skills": [],
                "technical_level": "Unknown",
                "interview_score": average_score,
                "job_category_preference": job_preference
            }
        
        # Update candidate score and status
        candidate_service.update_candidate_score(
            candidate_id=st.session_state['candidate_id'],
            **results_data
        )
        st.success("✅ Interview results and essential skill data saved successfully!")
    except Exception as e:
        st.error(f"❌ Error saving interview results: {str(e)}")

if __name__ == "__main__":
    main()

