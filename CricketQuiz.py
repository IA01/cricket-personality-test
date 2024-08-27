import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from mixpanel import Mixpanel
import uuid
import os

ATTRIBUTES = ['Personal Ambition', 'Team Player', 'Charismatic', 'Empathetic', 'Brilliance', 'Consistency', 'Work Ethic', 'Natural Ability']

def attribute_majority_match(user_profile, player_profiles):
    attribute_matches = {player: 0 for player in player_profiles}
    for attr in ATTRIBUTES:
        user_value = user_profile[attr]
        closest_players = []
        min_diff = float('inf')
        for player, profile in player_profiles.items():
            diff = abs(user_value - profile[attr])
            if diff < min_diff:
                closest_players = [player]
                min_diff = diff
            elif diff == min_diff:
                closest_players.append(player)
        for player in closest_players:
            attribute_matches[player] += 1
    return max(attribute_matches, key=attribute_matches.get)

def cosine_similarity(user_profile, player_profile):
    user_vector = np.array([user_profile[attr] for attr in ATTRIBUTES])
    player_vector = np.array([player_profile[attr] for attr in ATTRIBUTES])
    dot_product = np.dot(user_vector, player_vector)
    user_norm = np.linalg.norm(user_vector)
    player_norm = np.linalg.norm(player_vector)
    return dot_product / (user_norm * player_norm)

def cosine_match(user_profile, player_profiles):
    similarities = {player: cosine_similarity(user_profile, profile) for player, profile in player_profiles.items()}
    return max(similarities, key=similarities.get)

def manhattan_match(user_profile, player_profiles):
    distances = {player: sum(abs(user_profile[attr] - profile[attr]) for attr in ATTRIBUTES) 
                 for player, profile in player_profiles.items()}
    return min(distances, key=distances.get)

def voting_match(user_profile, player_profiles):
    methods = [attribute_majority_match, cosine_match, manhattan_match]
    votes = {}
    rankings = {player: 0 for player in player_profiles}

    for method_func in methods:
        best_match = method_func(user_profile, player_profiles)
        votes[best_match] = votes.get(best_match, 0) + 1
        
        if method_func == cosine_match:
            sorted_players = sorted(player_profiles.keys(), 
                                    key=lambda p: cosine_similarity(user_profile, player_profiles[p]), reverse=True)
        elif method_func == manhattan_match:
            sorted_players = sorted(player_profiles.keys(), 
                                    key=lambda p: sum(abs(user_profile[attr] - player_profiles[p][attr]) for attr in ATTRIBUTES))
        else:
            sorted_players = sorted(player_profiles.keys(), 
                                    key=lambda p: sum(1 for attr in ATTRIBUTES if abs(user_profile[attr] - player_profiles[p][attr]) == min(abs(user_profile[attr] - profile[attr]) for profile in player_profiles.values())), reverse=True)
        
        for rank, player in enumerate(sorted_players):
            rankings[player] += rank

    max_votes = max(votes.values())
    best_matches = [player for player, vote_count in votes.items() if vote_count == max_votes]

    return min(best_matches, key=lambda p: rankings[p]) if len(best_matches) > 1 else best_matches[0]

def find_best_match(user_profile, player_profiles):
    return voting_match(user_profile, player_profiles)

# Define questions (you'll need to add all 16 questions)
QUESTIONS = [
    {
        "text": "I would rather be a key player on a mediocre team than a bench player on a championship-winning side.",
        "attribute": "Personal Ambition",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "I'd rather be remembered as a winner than a nice person.",
        "attribute": "Personal Ambition",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "Collaboration gets the better out of me than competition.",
        "attribute": "Team Player",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "I'm more comfortable leading from behind the scenes than being the face of a team.",
        "attribute": "Team Player",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "When the stakes are high, I prefer taking the onus at the risk of being wrong than trusting someone else.",
        "attribute": "Charismatic",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "A team can have only one leader and it is usually me.",
        "attribute": "Charismatic",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "I occasionally downplay my achievements to avoid making others feel inadequate.",
        "attribute": "Empathetic",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "A team never loses because of an individual's fault. It is always collective failure.",
        "attribute": "Empathetic",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "I would rather face and overcome my biggest weaknesses than further enhance my greatest strengths.",
        "attribute": "Work Ethic",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "I strive to specialize and master one area rather than being a jack-of-all-trades.",
        "attribute": "Work Ethic",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "I sometimes find it hard to maintain focus during less exciting or routine tasks because I am able to do them easily.",
        "attribute": "Natural Ability",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "Before a big day, you will find me taking it easy and visualizing success than practicing or further fine-tuning my skills.",
        "attribute": "Natural Ability",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "I tend to trust my intuition even when it contradicts logical evidence.",
        "attribute": "Brilliance",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "If I had $50, I would wager it to win 100 even at the cost of losing the 50.",
        "attribute": "Brilliance",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "In crucial moments, I prefer to stick to tried-and-true methods rather than attempt something different.",
        "attribute": "Consistency",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    },
    {
        "text": "I work better with a plan and structure rather than in ambiguity and chaos.",
        "attribute": "Consistency",
        "weights": {"Strongly Disagree": 0.2, "Disagree": 0.4, "Neutral": 0.6, "Agree": 0.8, "Strongly Agree": 1.0}
    }
]

# Define player profiles
PLAYER_PROFILES = {
    "Sachin Tendulkar": {
        'Personal Ambition': 0.8, 'Team Player': 0.9,
        'Charismatic': 0.6, 'Empathetic': 0.8,
        'Brilliance': 0.9, 'Consistency': 0.9,
        'Work Ethic': 0.7, 'Natural Ability': 1.0
    },
    "Virat Kohli": {
        'Personal Ambition': 0.9, 'Team Player': 0.9,
        'Charismatic': 0.9, 'Empathetic': 0.4,
        'Brilliance': 0.8, 'Consistency': 1.0,
        'Work Ethic': 1.0, 'Natural Ability': 0.7
    },
    "Rahul Dravid": {
        'Personal Ambition': 0.8, 'Team Player': 1.0,
        'Charismatic': 0.4, 'Empathetic': 0.9,
        'Brilliance': 0.7, 'Consistency': 0.9,
        'Work Ethic': 0.9, 'Natural Ability': 0.8
    },
    "Rohit Sharma": {
        'Personal Ambition': 0.8, 'Team Player': 0.9,
        'Charismatic': 0.4, 'Empathetic': 0.9,
        'Brilliance': 0.9, 'Consistency': 0.6,
        'Work Ethic': 0.5, 'Natural Ability': 0.95
    },
    "MS Dhoni": {
        'Personal Ambition': 0.8, 'Team Player': 0.9,
        'Charismatic': 0.9, 'Empathetic': 0.9,
        'Brilliance': 0.9, 'Consistency': 0.8,
        'Work Ethic': 0.8, 'Natural Ability': 0.8
    },
    "Ravichandran Ashwin": {
        'Personal Ambition': 0.9, 'Team Player': 0.7,
        'Charismatic': 0.8, 'Empathetic': 0.8,
        'Brilliance': 0.8, 'Consistency': 0.9,
        'Work Ethic': 0.9, 'Natural Ability': 0.8
    },
    "Suryakumar Yadav": {
        'Personal Ambition': 0.7, 'Team Player': 0.9,
        'Charismatic': 0.7, 'Empathetic': 0.9,
        'Brilliance': 1.0, 'Consistency': 0.9,
        'Work Ethic': 0.8, 'Natural Ability': 1.0
    },
    "Gautam Gambhir": {
        'Personal Ambition': 0.8, 'Team Player': 0.8,
        'Charismatic': 0.9, 'Empathetic': 0.3,
        'Brilliance': 0.5, 'Consistency': 0.8,
        'Work Ethic': 0.9, 'Natural Ability': 0.6
    },
    "Virender Sehwag": {
        'Personal Ambition': 0.6, 'Team Player': 0.8,
        'Charismatic': 0.8, 'Empathetic': 0.7,
        'Brilliance': 1.0, 'Consistency': 0.8,
        'Work Ethic': 0.7, 'Natural Ability': 1.0
    },
    "Yuvraj Singh": {
        'Personal Ambition': 0.9, 'Team Player': 0.8,
        'Charismatic': 0.9, 'Empathetic': 0.6,
        'Brilliance': 0.9, 'Consistency': 0.8,
        'Work Ethic': 0.4, 'Natural Ability': 0.9
    },
    "Ricky Ponting": {
        'Personal Ambition': 1.0, 'Team Player': 0.6,
        'Charismatic': 1.0, 'Empathetic': 0.3,
        'Brilliance': 0.7, 'Consistency': 0.8,
        'Work Ethic': 0.9, 'Natural Ability': 0.8
    },
    "Shane Warne": {
        'Personal Ambition': 0.9, 'Team Player': 0.7,
        'Charismatic': 1.0, 'Empathetic': 0.2,
        'Brilliance': 0.9, 'Consistency': 0.9,
        'Work Ethic': 0.5, 'Natural Ability': 0.9
    },
    "Pat Cummins": {
        'Personal Ambition': 0.7, 'Team Player': 0.9,
        'Charismatic': 1.0, 'Empathetic': 0.9,
        'Brilliance': 0.7, 'Consistency': 1.0,
        'Work Ethic': 0.9, 'Natural Ability': 0.8
    },
    "Steven Smith": {
        'Personal Ambition': 0.8, 'Team Player': 0.8,
        'Charismatic': 0.7, 'Empathetic': 0.8,
        'Brilliance': 0.9, 'Consistency': 0.9,
        'Work Ethic': 0.9, 'Natural Ability': 0.8
    },
    "Glenn Maxwell": {
        'Personal Ambition': 0.8, 'Team Player': 0.9,
        'Charismatic': 0.8, 'Empathetic': 0.7,
        'Brilliance': 1.0, 'Consistency': 0.3,
        'Work Ethic': 0.7, 'Natural Ability': 0.9
    },
    "Ben Stokes": {
        'Personal Ambition': 0.8, 'Team Player': 1.0,
        'Charismatic': 1.0, 'Empathetic': 0.9,
        'Brilliance': 0.9, 'Consistency': 0.7,
        'Work Ethic': 0.9, 'Natural Ability': 0.8
    },
    "Eoin Morgan": {
        'Personal Ambition': 0.7, 'Team Player': 1.0,
        'Charismatic': 0.8, 'Empathetic': 1.0,
        'Brilliance': 0.4, 'Consistency': 0.7,
        'Work Ethic': 0.8, 'Natural Ability': 0.7
    },
    "Joe Root": {
        'Personal Ambition': 0.8, 'Team Player': 0.8,
        'Charismatic': 0.7, 'Empathetic': 0.8,
        'Brilliance': 0.8, 'Consistency': 1.0,
        'Work Ethic': 0.9, 'Natural Ability': 0.8
    },
    "Kevin Pietersen": {
        'Personal Ambition': 0.9, 'Team Player': 0.6,
        'Charismatic': 1.0, 'Empathetic': 0.5,
        'Brilliance': 0.9, 'Consistency': 0.7,
        'Work Ethic': 0.8, 'Natural Ability': 0.9
    },
    "Brendon McCullum": {
        'Personal Ambition': 0.6, 'Team Player': 0.9,
        'Charismatic': 0.9, 'Empathetic': 0.9,
        'Brilliance': 1.0, 'Consistency': 0.7,
        'Work Ethic': 0.8, 'Natural Ability': 0.8
    },
    "Kane Williamson": {
        'Personal Ambition': 0.8, 'Team Player': 0.9,
        'Charismatic': 0.4, 'Empathetic': 0.9,
        'Brilliance': 0.6, 'Consistency': 0.9,
        'Work Ethic': 0.9, 'Natural Ability': 0.8
    },
    "AB de Villiers": {
        'Personal Ambition': 0.7, 'Team Player': 0.8,
        'Charismatic': 0.8, 'Empathetic': 0.7,
        'Brilliance': 1.0, 'Consistency': 0.9,
        'Work Ethic': 0.9, 'Natural Ability': 1.0
    },
    
}

def calculate_user_profile(user_answers):
    profile = {attr: 0 for attr in ATTRIBUTES}
    question_count = {attr: 0 for attr in ATTRIBUTES}

    for answer, question in zip(user_answers, QUESTIONS):
        attr = question['attribute']
        if answer in question['weights']:
            profile[attr] += question['weights'][answer]
            question_count[attr] += 1

    # Normalize the scores and round to 2 decimal places
    for attr in ATTRIBUTES:
        if question_count[attr] > 0:
            profile[attr] = round(profile[attr] / question_count[attr], 2)
        else:
            del profile[attr]  # Remove attributes that weren't used

    return profile

# Initialize Mixpanel (add this after your existing imports and global variables)
mp = Mixpanel(os.environ.get("MIXPANEL_TOKEN", "e42e1cfb4c9c45d911dcd1156625e7ed"))
session_id = str(uuid.uuid4())

# Add this function to your existing code
def track_event(event_name, properties=None):
    if properties is None:
        properties = {}
    mp.track(session_id, event_name, properties)

def main():
    st.set_page_config(page_title="Cricket Personality Test", page_icon="🏏")
    
    # Track app visit
    track_event("App Visit")
    
    # Title
    st.markdown("<h1 style='text-align: center;'>Cricket Personality Test</h1>", unsafe_allow_html=True)

   
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = []

    if st.session_state.current_question < len(QUESTIONS):
        question = QUESTIONS[st.session_state.current_question]
        st.markdown(f"<div class='custom-header'>Question {st.session_state.current_question + 1} of {len(QUESTIONS)}</div>", unsafe_allow_html=True)
        st.write(question["text"])
        
        # Improve font of slider labels
        st.markdown("""
        <style>
        .stSlider [data-baseweb="typography"] {
            font-size: 1.1rem !important;
            font-weight: 500 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
          answer = st.select_slider(
            "",
            options=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
            value="Neutral",
            key=f"slider_{st.session_state.current_question}"
        )
        
        # Add Next button at the center
        # This code creates a centered button for navigating through the quiz
       
        # Determine if it's the last question to set appropriate button text
        button_text = "Finish" if st.session_state.current_question == len(QUESTIONS) - 1 else "Next"
        
        # Create a button with dynamic text
        if st.button(button_text):
            # When clicked:
                # 1. Save the user's answer
                st.session_state.user_answers.append(answer)
                # 2. Move to the next question
                st.session_state.current_question += 1
                # 3. Track the event for analytics
                track_event("Question Attempted", {"question_number": st.session_state.current_question})
                # 4. Refresh the page to show the next question or results
                st.rerun()

    else:
        user_profile = calculate_user_profile(st.session_state.user_answers)
        best_match = find_best_match(user_profile, PLAYER_PROFILES)
        display_results(user_profile)

        track_event("Quiz Completed", {
            "best_match": best_match,
            "questions_answered": len(st.session_state.user_answers)
        })

        if st.button("Play Again"):
            st.session_state.current_question = 0
            st.session_state.user_answers = []
            track_event("Quiz Restart")
            st.rerun()

    # Display progress bar
    progress = st.session_state.current_question / len(QUESTIONS)
    st.progress(progress)

def display_results(user_profile):
    st.subheader("Your Cricket Personality")

    best_match = find_best_match(user_profile, PLAYER_PROFILES)
    st.write(f"Your cricket personality most closely matches: **{best_match}**")

    # Explanation
    explanation = (
        f"Your profile aligns with {best_match} due to the similar balance of attributes in your personalities. "
        f"This doesn't mean your individual trait scores are identical to {best_match}'s, "
        f"but rather that the overall pattern of your strengths and characteristics is most similar to theirs. "
        f"The radar chart below shows your unique profile, highlighting your personal strengths and areas for growth."
    )
    st.write(explanation)

    # Radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    values = [user_profile.get(attr, 0) for attr in ATTRIBUTES]
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(ATTRIBUTES), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, ATTRIBUTES)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    
    plt.title("Your Cricket Personality Profile", y=1.08)
    st.pyplot(fig)

    # Additional insights
    
if __name__ == "__main__":
    main()
