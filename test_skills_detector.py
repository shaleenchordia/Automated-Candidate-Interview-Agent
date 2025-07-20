#!/usr/bin/env python3
"""
Test script for the expanded skill detection system
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.ml.inference.skills_detector import SkillDetector

def test_skills_detection():
    """Test the skill detection system with sample resume text"""
    
    # Initialize the detector
    detector = SkillDetector()
    
    # Sample resume text with various skills
    sample_resume = """
    John Doe
    Senior Software Engineer
    
    EXPERIENCE:
    Senior Full Stack Developer (5 years)
    - Developed web applications using Python, Django, and React
    - Built machine learning models with TensorFlow and PyTorch for computer vision tasks
    - Designed microservices architecture using Docker and Kubernetes
    - Implemented CI/CD pipelines with Jenkins and GitHub Actions
    - Worked with AWS services including EC2, S3, Lambda, and RDS
    - Led a team of 4 developers and mentored junior engineers
    - Used SQL databases (PostgreSQL) and NoSQL (MongoDB, Redis)
    - Implemented GraphQL APIs and REST services
    - Applied DevOps practices with Terraform and Ansible
    - Experience with blockchain development using Solidity and Ethereum
    - Conducted penetration testing and security audits
    - Performed load testing with JMeter and implemented monitoring with Prometheus
    
    SKILLS:
    - Programming: Python, JavaScript, TypeScript, Java, Go, Rust
    - Machine Learning: TensorFlow, PyTorch, scikit-learn, computer vision, NLP
    - Frontend: React, Vue.js, Angular, HTML, CSS, Tailwind CSS
    - Backend: Django, Flask, FastAPI, Express.js, Spring Boot
    - Cloud: AWS, Azure, GCP, Docker, Kubernetes
    - Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
    - DevOps: Jenkins, GitHub Actions, Terraform, Ansible
    - Testing: Unit testing, integration testing, Selenium, Cypress
    - Mobile: React Native, Flutter, iOS (Swift), Android (Kotlin)
    - Blockchain: Solidity, Web3, Ethereum, smart contracts
    - Data: Apache Spark, Kafka, Hadoop, data pipelines, ETL
    - Security: OWASP, penetration testing, encryption, OAuth
    - Project Management: Agile, Scrum, Jira, technical leadership
    
    EDUCATION:
    M.S. Computer Science
    B.S. Software Engineering
    
    CERTIFICATIONS:
    - AWS Certified Solutions Architect
    - Certified Scrum Master
    - TensorFlow Developer Certificate
    """
    
    print("🔍 Analyzing sample resume with expanded skill detection...")
    print("=" * 60)
    
    # Analyze the resume
    results = detector.analyze_resume(sample_resume)
    
    # Print overall summary
    print(f"📊 OVERALL ANALYSIS:")
    print(f"   Total skill categories detected: {results['total_categories_detected']}")
    print(f"   Overall technical level: {results['overall_technical_level']}")
    print(f"   Summary: {results['summary']}")
    print()
    
    # Print top skills
    print(f"🏆 TOP {min(15, len(results['top_skills']))} SKILLS:")
    for i, skill in enumerate(results['top_skills'][:15], 1):
        confidence = results['skill_scores'][skill]['confidence']
        level = results['skill_scores'][skill]['experience_level']
        evidence_count = results['skill_scores'][skill]['keyword_count']
        print(f"   {i:2d}. {skill:<30} | Confidence: {confidence:.3f} | Level: {level:<10} | Evidence: {evidence_count}")
    print()
    
    # Print detailed breakdown by domain
    print("📋 DETAILED BREAKDOWN BY DOMAIN:")
    print("-" * 60)
    
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
        ],
        'Security & Testing': [
            'Cybersecurity', 'Cloud Security', 'Application Security', 'Software Testing',
            'Performance Testing', 'Test Management'
        ],
        'Specialized Domains': [
            'Game Development', 'Blockchain & Web3', 'IoT & Embedded Systems',
            'Robotics & Automation', 'Fintech & Finance', 'Healthcare & Biotech'
        ],
        'Business & Management': [
            'Project Management', 'Technical Leadership', 'Communication & Collaboration',
            'Product Management', 'Business & Strategy', 'Sales & Marketing'
        ]
    }
    
    for domain_name, categories in domains.items():
        found_skills = []
        for category in categories:
            if category in results['skill_scores']:
                confidence = results['skill_scores'][category]['confidence']
                level = results['skill_scores'][category]['experience_level']
                found_skills.append((category, confidence, level))
        
        if found_skills:
            print(f"\n🔹 {domain_name.upper()}:")
            for skill, confidence, level in sorted(found_skills, key=lambda x: x[1], reverse=True):
                subcategories = results['skill_scores'][skill]['subcategories']
                evidence = results['skill_scores'][skill]['evidence'][:5]  # Show first 5 pieces of evidence
                print(f"   ✓ {skill:<35} | {confidence:.3f} | {level:<10}")
                print(f"     Subcategories: {', '.join(subcategories)}")
                print(f"     Evidence: {', '.join(evidence)}")
    
    print("\n" + "=" * 60)
    print("✅ Skill detection analysis complete!")
    
    return results

if __name__ == "__main__":
    test_skills_detection()
