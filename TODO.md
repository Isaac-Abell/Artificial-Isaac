# Artificial Isaac - Knowledge Base TODOs

This file contains the questions for the RAG Survey App. 
Answering these will populate your AI's long-term memory.

## 1. Identity & Biography
- [ ] **Basics:**
    - Full Name?
    - Preferred Name/Nickname?
    - Current Location (City, Country)?
    - Hometown? (likes, dislikes)
    - Age/Birthdate?
- **Background:**
    - Education History? (Degrees, Schools, Major)
    - Key Childhood Memories?
    - Significant Life Events?
- **Relationships:**
    - Relationship Status?
    - Significant Other's Name?
    - Family Members (Parents, Siblings)?
    - Best Friends?
    - Pets (Name, Type, Breed, Personality)?

## 2. Professional & Career
- [ ] **Current Role:**
    - Job Title?
    - Company/Industry?
    - Key Responsibilities?
    - Do you like your job? (Why/Why not?)
- **Career History:**
    - Past Companies/Roles?
    - Significant Achievements? (Projects, Awards, etc.)
    - Worst Job Experience? (Why was it bad?)
- **Work Style:**
    - Preferred Work Environment (Remote, Office, Hybrid)? (Why?)
    - Work Hours Preference? (why)
    - Leadership Style? (why)
    - Career Goals/Ambitions? (Where do you see yourself in 5 years?)

## 3. Skills & Expertise
- [ ] **Core Competencies:**
    - What are your top 3 professional skills? (why)
    - What are you "known for" among colleagues/friends?
    - Any hidden talents? (Juggling, Magic, Speed-reading?)
- **Skills & Tools:**
    - List your specific skills/proficiencies (Software, Languages, Equipment, etc.)?
    - Any strong opinions on specific methods or tools? (e.g. Tabs vs Spaces context)
    - Workflow Preferences?
- **Learning:**
    - What are you currently learning? (why)
    - What skill do you WISH you had? (why)
    - How do you learn best? (Books, Videos, Google, Doing it?)

## 4. Interests & Hobbies
- [ ] **Sports/Fitness:**
    - Favorite Sport(s) to Watch? (why)
    - Favorite Team(s)? (why)
    - Favorite Athlete(s)? (why)
    - Do you play any sports?
        - If yes: Which ones?
        - Skill Level?
        - Specific Gear/Equipment? (why you use it)
    - Workout Routine (Gym, Running, Yoga, etc.)?
- **Gaming:**
    - Do you play video games?
        - Favorite Genres (FPS, RPG, RTS)?
        - Specific Titles (LoL, COD, Elden Ring, etc.)? (why)
        - Console vs. PC?
        - Competitive or Casual?
        - In-game Handle/Username?
- **Media:**
    - Favorite Music Genres/Artists? (why)
    - Favorite Movies? (why)
    - Favorite TV Shows? (why)
    - Currently Reading/Listening to? (why)
    - Favorite Content Creators/YouTubers? (why)
- **Food & Drink:**
    - Favorite Foods?
    - Favorite Drinks (Alcoholic/Non-Alcoholic)?
    - Dietary Restrictions/Preferences?
    - Can you cook? Signature Dish?

## 5. Personality & Opinions
- [ ] **Style:**
    - Introvert or Extrovert?
    - Optimist or Pessimist?
    - Morning Person or Night Owl?
    - Clean or Messy?
- **Politics/Philosophy:**
    - General Political Leaning?
    - Religious/Spiritual Beliefs?
    - Controversial Opinions?
    - Core Values?
- **Communication:**
    - Use of Slang/Emojis?
    - Favorite Catchphrases?
    - Sarcastic or Serious?
    - Texting Style (Short/Long, Fast/Slow)?

## 6. Travel & Experiences
- [ ] **Travel:**
    - Favorite Place You've Visited?
    - Meaningful Trips?
    - Dream Destination?
    - Travel Style (Backpacking, Luxury, Adventure)?
- **Bucket List:**
    - Top 3 Things to Do Before You Die?

---

## 7. Future RAG Format (Simplification Plan)

**Proposed Format (JSON - Q&A Style):**

File: `rag_data/biography.json`
```json
[
  {
    "question": "Where did you grow up?",
    "answer": "I grew up in Seattle, WA. I loved the rain but hated the traffic."
  },
  {
    "question": "What is your biggest fear?",
    "answer": "Spiders and public speaking."
  }
]
```

**Why this is better:**
*   **Predictable:** No guessing what a nested dictionary key means.
*   **Retrievable:** Each Q&A pair or Markdown section is a perfect, self-contained chunk for the LLM to read.

