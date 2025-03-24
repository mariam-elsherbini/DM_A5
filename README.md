# LLM Workflow Implementation

## Overview
This assignment implements various LLM workflow approaches for content repurposing. The workflows take a long-form blog post and transform it into different formats, including a summary, social media posts, and an email newsletter. The project includes:
- A **Pipeline Workflow**
- A **DAG Workflow with Reflexion (Self-Correction)**
- An **Agent-Driven Workflow**

Additionally, a comparative evaluation is conducted to analyze the effectiveness of each approach.

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- `pip` package manager
- OpenAI or Groq API keys
- `.env` file configured with API keys and model settings

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up API keys in a `.env` file:
   ```env
   MODEL_SERVER=GROQ  # or OPENAI
   GROQ_API_KEY=your_api_key_here
   GROQ_BASE_URL=your_groq_url_here
   GROQ_MODEL=your_model_name_here
   ```
   Modify the variables based on your model provider.
5. Run the workflow script:
   ```bash
   python llm_workflow.py
   ```

---

## Implementation Overview

### Workflow Approaches

#### 1. **Pipeline Workflow**
A sequential approach where each task's output feeds into the next task.
- Extracts key points from a blog post.
- Generates a summary from key points.
- Creates social media posts.
- Forms an email newsletter.

#### 2. **DAG Workflow with Reflexion**
A more flexible workflow where tasks can have multiple dependencies.
- Uses Reflexion to evaluate and improve generated content iteratively.
- Enhances content generation quality.

#### 3. **Agent-Driven Workflow**
A dynamic workflow where an AI agent decides the execution order of tasks.
- Uses an autonomous decision-making loop.
- Integrates feedback to optimize content generation.

---

## Example Outputs

### **Pipeline Workflow Output:**
```json
{
  "summary": "AI is transforming healthcare by improving diagnostics, optimizing treatment plans, and enabling personalized medicine.",
  "social_media": {
    "twitter": "AI in healthcare is revolutionizing diagnostics and treatment! #AI #HealthcareTech",
    "linkedin": "AI is reshaping the healthcare industry with advanced diagnostics and personalized medicine. Learn more!",
    "facebook": "How AI is making healthcare smarter and more efficient. Read more!"
  },
  "email_newsletter": {
    "subject": "The Future of AI in Healthcare",
    "body": "Discover how AI is transforming healthcare by enhancing diagnostics and optimizing treatment plans."
  }
}
```

### **DAG Workflow Output with Reflexion:**
```json
{
  "summary": "With AI, healthcare professionals can diagnose diseases earlier, customize treatments, and reduce operational costs.",
  "social_media": {
    "twitter": "AI-driven diagnostics are improving patient outcomes! Learn more about AI in healthcare. #AI #HealthTech",
    "linkedin": "AI is revolutionizing healthcare by providing early disease detection and customized treatment plans. Explore the impact!",
    "facebook": "AI is changing the healthcare landscape with better diagnostics and cost reduction. Read about it here."
  },
  "email_newsletter": {
    "subject": "AI and the Future of Medicine",
    "body": "The integration of AI in healthcare is bringing new opportunities for better diagnosis and personalized treatments."
  }
}
```

### **Agent-Driven Workflow Output:**
```json
{
  "summary": "AI in healthcare is improving patient care by optimizing diagnostics, reducing costs, and personalizing treatments.",
  "social_media": {
    "twitter": "The future of healthcare is AI-powered! Smarter diagnostics and personalized medicine are here. #AI #HealthTech",
    "linkedin": "AI's impact on healthcare is profound: reducing costs, improving diagnosis, and personalizing treatments. Learn more!",
    "facebook": "AI is optimizing healthcare in ways we never imagined! Read about the transformation here."
  },
  "email_newsletter": {
    "subject": "AI is Transforming Healthcare",
    "body": "Learn how AI is improving patient care through personalized medicine, smart diagnostics, and cost reductions."
  }
}
```

---

## Workflow Effectiveness Analysis
| Workflow Type  | Pros | Cons |
|---------------|------|------|
| **Pipeline**  | Simple, easy to debug, structured | Rigid, limited adaptability |
| **DAG + Reflexion**  | Flexible, self-correcting, improves output quality | Higher complexity, requires more iterations |
| **Agent-Driven** | Adaptive, efficient, dynamically optimized | Requires fine-tuning, risk of inefficient decision loops |

### Key Takeaways
- **Pipeline workflow** is best for straightforward content repurposing tasks.
- **DAG with Reflexion** enhances quality but increases computational cost.
- **Agent-driven workflows** are highly flexible but require careful design to ensure efficiency.

---

## Challenges and Solutions

### **Challenge: Handling API Rate Limits**
- **Issue:** Frequent API calls could exceed rate limits.
- **Solution:** Implemented exponential backoff and retry logic.

### **Challenge: Maintaining Content Quality**
- **Issue:** LLM-generated content could lack coherence.
- **Solution:** Integrated self-correction mechanisms using Reflexion.

### **Challenge: Ensuring Agent Efficiency**
- **Issue:** The agent sometimes entered unnecessary decision loops.
- **Solution:** Capped iterations and enforced structured reasoning.

---

## Conclusion
This project explored multiple LLM workflow approaches for content repurposing. While **Pipeline workflows** are easy to implement, **DAG with Reflexion** improves quality, and **Agent-driven workflows** offer maximum flexibility. The comparative analysis provides insights into when to use each approach based on project needs.

---

## Future Improvements
- Implementing **parallel processing** for faster execution.
- Fine-tuning LLM models for domain-specific tasks.
- Exploring **multi-agent collaboration** for complex workflows.
