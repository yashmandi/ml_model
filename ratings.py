import re
import pickle
from collections import Counter

def assign_rating(resume_text):
    score = 0
    max_score = 20  # Increased maximum score for more granularity

    # Education (0-3 points)
    education_level = {
        "phd": 3,
        "master": 3,
        "masters": 3,
        "bachelor": 1
    }
    for degree, points in education_level.items():
        if re.search(rf"\b{degree}\b", resume_text, re.I):
            score += points
            break
    print(f"Education score: {score}")

    # Skills (0-4 points)
    tech_skills = ["python", "javascript", "java", "c++", "c#", "ruby", "go", "rust", 
                   "sql", "nosql", "react", "angular", "vue", "node.js", "django", 
                   "flask", "spring", "asp.net", "docker", "kubernetes", "aws", "azure", 
                   "gcp", "machine learning", "data analysis", "ai", "blockchain", "nlp",
                   "react", "node", "tailwind", "express", "react native", "redux", "graphql", "nginx", "apache"] 
    skills_count = sum(1 for skill in tech_skills if re.search(rf"\b{skill}\b", resume_text, re.I))
    score += min(skills_count, 8)
    print(f"Skills score: {score}")

    # Projects (0-2 points)
    project_count = len(re.findall(r"\b(project|project\s*experience)\b", resume_text, re.I))
    score += min(project_count, 4)
    print(f"Projects score: {score}")

    # Experience (0-3 points)
    years_exp = re.search(r"(\d+)\+?\s*years?\s*(?:of)?\s*experience", resume_text, re.I)
    if years_exp:
        years = int(years_exp.group(1))
        score += min(years // 2, 3)
    print(f"Experience score: {score}")

    # ATS keywords (0-3 points)
    ats_keywords = ["team player", "problem solver", "leadership", "communication", "collaboration", 
                "agile", "scrum", "devops", "ci/cd", "test-driven development", "rest api", 
                "microservices", "data structures", "algorithms", "object-oriented programming", 
                "functional programming", "version control", "git", "code review", "debugging", 
                "stakeholder management", "time management", "project management", "critical thinking",
                "adaptability", "innovation", "strategic planning", "customer service", 
                "cross-functional teams", "design patterns", "system architecture", 
                "cloud computing", "business analysis", "data visualization", "big data",
                "cybersecurity", "networking", "IT infrastructure", "full stack development", 
                "UI/UX design", "user research", "product management", "data mining", 
                "ETL processes", "business intelligence", "data warehousing", "data governance"]

    ats_count = sum(1 for keyword in ats_keywords if re.search(rf"\b{keyword}\b", resume_text, re.I))
    score += min(ats_count // 2, 3)
    print(f"ATS keywords score: {score}")

    # Certifications (0-2 points)
    certifications = ["aws certified", "microsoft certified", "google certified", "cisco certified", 
                      "comptia", "pmp", "scrum", "itil", "cissp", "ceh"]
    cert_count = sum(1 for cert in certifications if re.search(rf"\b{cert}\b", resume_text, re.I))
    score += min(cert_count, 2)
    print(f"Certifications score: {score}")

    # Achievements and Awards (0-2 points)
    achievements = ["award", "recognition", "honor", "prize", "scholarship", "fellowship"]
    achievement_count = sum(1 for ach in achievements if re.search(rf"\b{ach}\b", resume_text, re.I))
    score += min(achievement_count, 7)
    print(f"Achievements score: {score}")

    # Quantifiable results (0-1 point)
    if re.search(r"\b(increased|decreased|improved|reduced|saved|generated)\b.*\d+%", resume_text, re.I):
        score += 1
    print(f"Quantifiable results score: {score}")

    # Normalize score to be out of 10
    rating = (score / max_score) * 10
    rounded_rating = round(rating * 2) / 2  # Round to nearest 0.5
    print(f"Final rating: {rounded_rating}")

    return rounded_rating

# Load preprocessed texts
with open('preprocessed_texts.pkl', 'rb') as f:
    preprocessed_texts = pickle.load(f)

ratings = [assign_rating(text) for text in preprocessed_texts]

# Save ratings
with open('ratings.pkl', 'wb') as f:
    pickle.dump(ratings, f)

# Print distribution of ratings
rating_counts = Counter(ratings)
print("Rating distribution:")
for rating in sorted(rating_counts.keys()):
    print(f"Rating {rating}: {rating_counts[rating]} resumes")