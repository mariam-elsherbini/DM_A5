import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the model server type
model_server = os.getenv('MODEL_SERVER', 'GROQ').upper() # Default to GROQ if not set
if model_server == "OPTOGPT":
    API_KEY = os.getenv('OPTOGPT_API_KEY')
    BASE_URL = os.getenv('OPTOGPT_BASE_URL')
    LLM_MODEL = os.getenv('OPTOGPT_MODEL')
elif model_server == "GROQ":
    API_KEY = os.getenv('GROQ_API_KEY')
    BASE_URL = os.getenv('GROQ_BASE_URL')
    LLM_MODEL = os.getenv('GROQ_MODEL')
elif model_server == "NGU":
    API_KEY = os.getenv('NGU_API_KEY')
    BASE_URL = os.getenv('NGU_BASE_URL')
    LLM_MODEL = os.getenv('NGU_MODEL')
elif model_server == "OPENAI":
    API_KEY = os.getenv('OPENAI_API_KEY')
    BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1') # Default to OpenAI's standard base URL
    LLM_MODEL = os.getenv('OPENAI_MODEL')
else:
    raise ValueError(f"Unsupported MODEL_SERVER: {model_server}")

# Initialize the OpenAI client with custom base URL
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Define a function to make LLM API calls
def call_llm(messages, tools=None, tool_choice=None):
    """
    Make a call to the LLM API with the specified messages and tools.
    Args:
        messages: List of message objects
        tools: List of tool definitions (optional)
        tool_choice: Tool choice configuration (optional)
    Returns:
        The API response
    """
    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    try:
        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None

def get_sample_blog_post():
    """
    Read the sample blog post from a JSON file.
    """
    try:
        with open('sample_blog_post.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: sample_blog_post.json file not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in sample_blog_post.json.")
        return None

# Define tool schemas for each task
extract_key_points_schema = {
    "type": "function",
    "function": {
        "name": "extract_key_points",
        "description": "Extract key points from a blog post",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the blog post"
                },
                "content": {
                    "type": "string",
                    "description": "The content of the blog post"
                },
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of key points extracted from the blog post"
                }
            },
            "required": ["key_points"]
        }
    }
}

generate_summary_schema = {
    "type": "function",
    "function": {
        "name": "generate_summary",
        "description": "Generate a concise summary from the key points",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary of the blog post"
                }
            },
            "required": ["summary"]
        }
    }
}

create_social_media_posts_schema = {
    "type": "function",
    "function": {
        "name": "create_social_media_posts",
        "description": "Create social media posts for different platforms",
        "parameters": {
            "type": "object",
            "properties": {
                "twitter": {
                    "type": "string",
                    "description": "Post optimized for Twitter/X (max 280 characters)"
                },
                "linkedin": {
                    "type": "string",
                    "description": "Post optimized for LinkedIn (professional tone)"
                },
                "facebook": {
                    "type": "string",
                    "description": "Post optimized for Facebook"
                }
            },
            "required": ["twitter", "linkedin", "facebook"]
        }
    }
}

create_email_newsletter_schema = {
    "type": "function",
    "function": {
        "name": "create_email_newsletter",
        "description": "Create an email newsletter from the blog post and summary",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content in plain text"
                }
            },
            "required": ["subject", "body"]
        }
    }
}

def task_extract_key_points(blog_post):
    """
    Task function to extract key points from a blog post using tool calling.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        List of key points
    """
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content and extracting key points from articles."},
        {"role": "user", "content": f"Extract the key points from this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    
    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[extract_key_points_schema],
        tool_choice={"type": "function", "function": {"name": "extract_key_points"}}
    )
    
    # Extract the tool call information
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("key_points", [])
    return [] # Fallback if tool calling fails

def task_generate_summary(key_points, max_length=150):
    """
    Task function to generate a concise summary from key points using tool calling.
    Args:
        key_points: List of key points extracted from the blog post
        max_length: Maximum length of the summary in words
    Returns:
        String containing the summary
    """
    messages = [
        {"role": "system", "content": "You are an expert at summarizing content concisely while preserving key information."},
        {"role": "user", "content": f"Generate a summary based on these key points, max {max_length} words:\n\n" +
            "\n".join([f"- {point}" for point in key_points])}
    ]
    
    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[generate_summary_schema],
        tool_choice={"type": "function", "function": {"name": "generate_summary"}}
    )
    
    # Extract the tool call information
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("summary", "")
    return "" # Fallback if tool calling fails

def task_create_social_media_posts(key_points, blog_title):
    """
    Task function to create social media posts for different platforms using tool calling.
    Args:
        key_points: List of key points extracted from the blog post
        blog_title: Title of the blog post
    Returns:
        Dictionary with posts for each platform
    """
    messages = [
        {"role": "system", "content": "You are a social media expert who creates engaging posts optimized for different platforms."},
        {"role": "user", "content": f"Create social media posts for Twitter, LinkedIn, and Facebook based on this blog title: '{blog_title}' and these key points:\n\n" +
            "\n".join([f"- {point}" for point in key_points])}
    ]
    
    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[create_social_media_posts_schema],
        tool_choice={"type": "function", "function": {"name": "create_social_media_posts"}}
    )
    
    # Extract the tool call information
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    return {"twitter": "", "linkedin": "", "facebook": ""} # Fallback if tool calling fails

def task_create_email_newsletter(blog_post, summary, key_points):
    """
    Task function to create an email newsletter using tool calling.
    Args:
        blog_post: Dictionary containing the blog post
        summary: String containing the summary
        key_points: List of key points extracted from the blog post
    Returns:
        Dictionary with subject and body for the email newsletter
    """
    messages = [
        {"role": "system", "content": "You are an email marketing specialist who creates engaging newsletters."},
        {"role": "user", "content": f"Create an email newsletter based on this blog post:\n\nTitle: {blog_post['title']}\n\nSummary: {summary}\n\nKey Points:\n" +
            "\n".join([f"- {point}" for point in key_points])}
    ]
    
    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[create_email_newsletter_schema],
        tool_choice={"type": "function", "function": {"name": "create_email_newsletter"}}
    )
    
    # Extract the tool call information
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    return {"subject": "", "body": ""} # Fallback if tool calling fails

def run_pipeline_workflow(blog_post):
    """
    Run a simple pipeline workflow to repurpose content.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        Dictionary with all the generated content
    """
    print("Running pipeline workflow...")
    
    # Step 1: Extract key points from the blog post
    print("Step 1: Extracting key points...")
    key_points = task_extract_key_points(blog_post)
    
    # Step 2: Generate a summary from the key points
    print("Step 2: Generating summary...")
    summary = task_generate_summary(key_points)
    
    # Step 3: Create social media posts
    print("Step 3: Creating social media posts...")
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    
    # Step 4: Create an email newsletter
    print("Step 4: Creating email newsletter...")
    email = task_create_email_newsletter(blog_post, summary, key_points)
    
    # Return all generated content
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

def run_dag_workflow(blog_post):
    """
    Run a DAG workflow to repurpose content.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        Dictionary with all the generated content
    """
    print("Running DAG workflow...")
    
    # Step 1: Extract key points (this is a source node in the DAG)
    print("Node 1: Extracting key points...")
    key_points = task_extract_key_points(blog_post)
    
    # Step 2a and 2b can run in parallel as they both depend only on key_points
    # Step 2a: Generate summary
    print("Node 2a: Generating summary...")
    summary = task_generate_summary(key_points)
    
    # Step 2b: Create social media posts
    print("Node 2b: Creating social media posts...")
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    
    # Step 3: Email newsletter depends on blog_post, summary, and key_points
    print("Node 3: Creating email newsletter...")
    email = task_create_email_newsletter(blog_post, summary, key_points)
    
    # Return all generated content
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

def extract_key_points_with_cot(blog_post):
    """
    Extract key points from a blog post using chain-of-thought reasoning.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        List of key points
    """
    print("Extracting key points with chain-of-thought reasoning...")
    
    # Create a system message that instructs the model to use chain-of-thought
    messages = [
        {
            "role": "system", 
            "content": """You are an expert at analyzing content and extracting key points from articles.
            Use the following step-by-step approach:
            1. First, identify the major sections of the article
            2. For each section, identify the primary message or insight
            3. Consider which points are most important and impactful
            4. Determine which information would be most valuable to readers
            5. Finally, formulate a list of key points that capture the essence of the article
            
            Think through this process carefully before providing your final answer."""
        },
        {
            "role": "user", 
            "content": f"Extract the key points from this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"
        }
    ]
    
    # Two-step process: first get the chain-of-thought reasoning
    response = call_llm(messages=messages)
    
    # Extract the model's reasoning
    cot_reasoning = response.choices[0].message.content
    
    # Now extract structured key points using the tool
    messages.append({"role": "assistant", "content": cot_reasoning})
    messages.append({"role": "user", "content": "Now, based on your analysis, provide a concise list of key points from the article."})
    
    response = call_llm(
        messages=messages,
        tools=[extract_key_points_schema],
        tool_choice={"type": "function", "function": {"name": "extract_key_points"}}
    )
    
    # Extract the tool call information
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("key_points", [])
    
    return [] # Fallback if tool calling fails

def evaluate_content(content, content_type):
    """
    Evaluate the quality of generated content.
    Args:
        content: The content to evaluate
        content_type: The type of content (e.g., "summary", "social_media_post", "email")
    Returns:
        Dictionary with evaluation results and feedback
    """
    print(f"Evaluating {content_type}...")
    
    # Define evaluation criteria based on content type
    if content_type == "summary":
        criteria = """
        1. Conciseness: Is the summary brief but comprehensive?
        2. Clarity: Is the information presented clearly?
        3. Completeness: Does it capture all the main points?
        4. Coherence: Does it flow logically?
        5. Accuracy: Is the information accurate to the original?
        """
        content_description = "a summary of a blog post"
        
    elif content_type == "social_media_post":
        criteria = """
        1. Engagement: Is the post likely to engage the target audience?
        2. Clarity: Is the message clear and concise?
        3. Platform-Appropriateness: Does it follow the conventions of the platform?
        4. Call-to-Action: Does it include a compelling call to action?
        5. Value: Does it provide value to the audience?
        """
        content_description = "a social media post"
        
    elif content_type == "email":
        criteria = """
        1. Subject Line: Is the subject line compelling and accurate?
        2. Structure: Is the email well-structured?
        3. Content: Is the content valuable and relevant?
        4. Tone: Is the tone appropriate for the audience?
        5. Call-to-Action: Does it include a clear call to action?
        """
        content_description = "an email newsletter"
        
    else:
        criteria = """
        1. Clarity: Is the content clear and easy to understand?
        2. Relevance: Is the content relevant to the intended purpose?
        3. Structure: Is the content well-structured?
        4. Quality: Is the writing of high quality?
        5. Value: Does it provide value to the reader?
        """
        content_description = "a piece of content"
    
    # Prepare the prompt for evaluation
    messages = [
        {
            "role": "system",
            "content": f"You are an expert content evaluator. Your task is to evaluate {content_description} based on specific criteria."
        },
        {
            "role": "user",
            "content": f"""Please evaluate the following {content_type} based on these criteria:
            
            {criteria}
            
            For each criterion, assign a score from 1-10, where 1 is poor and 10 is excellent.
            Additionally, provide specific feedback on how to improve the content.
            
            Content to evaluate:
            {content}
            
            Provide your evaluation in JSON format with these fields:
            - criteria_scores: An object with each criterion as a key and the score as a value
            - overall_score: The overall quality score (average of all criteria scores), normalized to a 0-1 scale
            - feedback: Detailed feedback on how to improve the content
            """
        }
    ]
    
    # Get evaluation from the model
    response = call_llm(messages=messages)
    evaluation_text = response.choices[0].message.content
    
    # Extract JSON from the response
    try:
        # Try to find JSON in the response
        import re
        json_match = re.search(r'```json\n(.*?)\n```', evaluation_text, re.DOTALL)
        if json_match:
            evaluation_json = json.loads(json_match.group(1))
        else:
            # Try to parse the entire response as JSON
            evaluation_json = json.loads(evaluation_text)
        
        if "overall_score" not in evaluation_json:
            # Calculate the overall score from criteria_scores if available
            if "criteria_scores" in evaluation_json and evaluation_json["criteria_scores"]:
                scores = evaluation_json["criteria_scores"].values()
                evaluation_json["overall_score"] = sum(scores) / len(scores) / 10.0 #normalize to 0-1 scale
            else:
                # Provide a default overall score if not available
                evaluation_json["overall_score"] = 0.5 #default value
        if "feedback" not in evaluation_json:
            evaluation_json["feedback"] = "Improve clarity and content quality."
           

    except:
        # If JSON parsing fails, create a default structure
        print("Failed to parse evaluation as JSON. Creating default structure.")
        evaluation_json = {
            "criteria_scores": {},
            "overall_score": 0.5,
            "feedback": "Unable to parse detailed feedback."
        }
    
    return evaluation_json

def improve_content(content, feedback, content_type):
    """
    Improve content based on feedback.
    Args:
        content: The content to improve
        feedback: Feedback on how to improve the content
        content_type: The type of content
    Returns:
        Improved content
    """
    print(f"Improving {content_type} based on feedback...")
    
    # Define specific instructions based on content type
    if content_type == "summary":
        task_description = "summary of a blog post"
        tool_schema = generate_summary_schema
        tool_name = "generate_summary"
        
    elif content_type == "social_media_post":
        task_description = "social media post"
        tool_schema = create_social_media_posts_schema
        tool_name = "create_social_media_posts"
        
    elif content_type == "email":
        task_description = "email newsletter"
        tool_schema = create_email_newsletter_schema
        tool_name = "create_email_newsletter"
        
    else:
        task_description = "content"
        # Fallback to a generic approach for unknown content types
        messages = [
            {"role": "system", "content": "You are an expert content creator skilled at improving content based on feedback."},
            {"role": "user", "content": f"""Here is some content that needs improvement:
            
            {content}
            
            Here is feedback on how to improve it:
            
            {feedback}
            
            Please provide an improved version of the content based on this feedback."""}
        ]
        
        response = call_llm(messages=messages)
        return response.choices[0].message.content
    
    # For known content types, use tool calling for structured output
    messages = [
        {"role": "system", "content": f"You are an expert at creating high-quality {task_description}s."},
        {"role": "user", "content": f"""Here is a {task_description} that needs improvement:
        
        {content}
        
        Here is feedback on how to improve it:
        
        {feedback}
        
        Please provide an improved version of the {task_description} based on this feedback."""}
    ]
    
    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[tool_schema],
        tool_choice={"type": "function", "function": {"name": tool_name}}
    )
    
    # Extract the tool call information
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        
        # Handle different content types
        if content_type == "summary":
            return result.get("summary", "")
        elif content_type == "social_media_post":
            return result
        elif content_type == "email":
            return result
        
    # Fallback if tool calling fails
    return content

def generate_with_reflexion(generator_func, max_attempts=3):
    """
    Apply Reflexion to a content generation function.
    Args:
        generator_func: Function that generates content
        max_attempts: Maximum number of correction attempts
    Returns:
        Function that generates self-corrected content
    """
    def wrapped_generator(*args, **kwargs):
        # Get the content type from kwargs or use a default
        content_type = kwargs.pop("content_type", "content")
        
        # Generate initial content
        content = generator_func(*args, **kwargs)
        
        # Evaluate and correct if needed
        for attempt in range(max_attempts):
            print(f"Self-correction attempt {attempt + 1}/{max_attempts}")
            
            # Evaluate the content
            evaluation = evaluate_content(content, content_type)
            
            #check if evaluation has the required key, otherwise proide a default
            overall_score = evaluation.get("overall_score", 0.5)

            # If quality is good enough, return the content
            if overall_score >= 0.8: # Assuming score is between 0 and 1
                print(f"Content quality is good enough (score: {overall_score:.2f}). Returning content.")
                return content
                
            # Otherwise, attempt to improve the content
            print(f"Content quality needs improvement (score: {evaluation['overall_score']:.2f}). Applying feedback.")
            improved_content = improve_content(content, evaluation["feedback"], content_type)
            content = improved_content
            
        # Return the best content after max_attempts
        print(f"Finished self-correction after {max_attempts} attempts.")
        return content
        
    return wrapped_generator

def run_workflow_with_reflexion(blog_post):
    """
    Run a workflow with Reflexion-based self-correction.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        Dictionary with all the generated content
    """
    print("Running workflow with Reflexion-based self-correction...")
    
    # Apply Reflexion to each task
    extract_key_points_with_reflexion = generate_with_reflexion(
        task_extract_key_points, 
        max_attempts=2
    )
    
    generate_summary_with_reflexion = generate_with_reflexion(
        task_generate_summary, 
        max_attempts=2
    )
    
    create_social_media_posts_with_reflexion = generate_with_reflexion(
        task_create_social_media_posts, 
        max_attempts=2
    )
    
    create_email_newsletter_with_reflexion = generate_with_reflexion(
        task_create_email_newsletter, 
        max_attempts=2
    )
    
    # Step 1: Extract key points from the blog post
    print("Step 1: Extracting key points with self-correction...")
    key_points = extract_key_points_with_reflexion(blog_post, content_type="key_points")
    
    # Step 2: Generate a summary from the key points
    print("Step 2: Generating summary with self-correction...")
    summary = generate_summary_with_reflexion(key_points, content_type="summary")
    
    # Step 3: Create social media posts
    print("Step 3: Creating social media posts with self-correction...")
    social_posts = create_social_media_posts_with_reflexion(
        key_points, 
        blog_post['title'], 
        content_type="social_media_post"
    )
    
    # Step 4: Create an email newsletter
    print("Step 4: Creating email newsletter with self-correction...")
    email = create_email_newsletter_with_reflexion(
        blog_post, 
        summary, 
        key_points, 
        content_type="email"
    )
    
    # Return all generated content
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

def define_agent_tools():
    """
    Define the tools that the workflow agent can use.
    Returns:
        List of tool definitions
    """
    # Define the list of tools from Task 1.2
    all_tools = [
        extract_key_points_schema,
        generate_summary_schema,
        create_social_media_posts_schema,
        create_email_newsletter_schema
    ]
    
    # Add a "finish" tool
    finish_tool_schema = {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Complete the workflow and return the final results",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The final summary"
                    },
                    "social_posts": {
                        "type": "object",
                        "description": "The social media posts for each platform"
                    },
                    "email": {
                        "type": "object",
                        "description": "The email newsletter"
                    }
                },
                "required": ["summary", "social_posts", "email"]
            }
        }
    }
    
    # Return all tools, including the "finish" tool
    return all_tools + [finish_tool_schema]

def execute_agent_tool(tool_name, arguments):
    """
    Execute a tool based on the tool name and arguments.
    Args:
        tool_name: The name of the tool to execute
        arguments: The arguments to pass to the tool
    Returns:
        The result of executing the tool
    """
    print(f"Executing tool: {tool_name}")
    
    if tool_name == "extract_key_points":
        # Create a blog post structure if only content is provided
        if "content" in arguments and "title" in arguments:
            blog_post = {
                "title": arguments["title"],
                "content": arguments["content"]
            }
            return task_extract_key_points(blog_post)
        else:
            return {"error": "Missing required arguments: title and/or content"}
            
    elif tool_name == "generate_summary":
        if "key_points" in arguments:
            max_length = arguments.get("max_length", 150)
            return {"summary": task_generate_summary(arguments["key_points"], max_length)}
        else:
            return {"error": "Missing required argument: key_points"}
            
    elif tool_name == "create_social_media_posts":
        if "key_points" in arguments and "blog_title" in arguments:
            return task_create_social_media_posts(arguments["key_points"], arguments["blog_title"])
        else:
            return {"error": "Missing required arguments: key_points and/or blog_title"}
            
    elif tool_name == "create_email_newsletter":
        if "blog_post" in arguments and "summary" in arguments and "key_points" in arguments:
            return task_create_email_newsletter(
                arguments["blog_post"], 
                arguments["summary"], 
                arguments["key_points"]
            )
        else:
            return {"error": "Missing required arguments: blog_post, summary, and/or key_points"}
            
    elif tool_name == "finish":
        # The finish tool doesn't execute anything, just returns the provided content
        return arguments
        
    else:
        return {"error": f"Unknown tool: {tool_name}"}

def run_agent_workflow(blog_post):
    """
    Run an agent-driven workflow to repurpose content.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        Dictionary with all the generated content
    """
    print("Running agent-driven workflow...")
    
    # Define the system message for the agent
    system_message = """
    You are a Content Repurposing Agent. Your job is to take a blog post and repurpose it into different formats:
    1. Extract key points from the blog post
    2. Generate a concise summary
    3. Create social media posts for different platforms
    4. Create an email newsletter
    
    You have access to tools that can help you with each of these tasks. Think carefully about which tools to use and in what order.
    
    When you're done, use the 'finish' tool to complete the workflow.
    """
    
    # Initialize the conversation
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Please repurpose this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    
    # Define the agent tools
    tools = define_agent_tools()
    
    # Keep track of the results
    results = {}
    
    # Run the agent workflow
    max_iterations = 10
    for i in range(max_iterations):
        print(f"Agent iteration {i + 1}/{max_iterations}")
        
        # Get the agent's next action
        response = call_llm(messages, tools)
        
        # Add the agent's response to the conversation
        messages.append(response.choices[0].message)
        
        # Check if the agent is done
        if not response.choices[0].message.tool_calls:
            print("Agent finished without using tools.")
            break
            
        # Process the tool calls
        for tool_call in response.choices[0].message.tool_calls:
            # Extract tool information
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            # Check if the agent is done
            if tool_name == "finish":
                print("Agent has completed the workflow.")
                return arguments
                
            # Execute the tool
            tool_result = execute_agent_tool(tool_name, arguments)
            
            # Store results for use in other tool calls
            if tool_name == "extract_key_points":
                results["key_points"] = tool_result
            elif tool_name == "generate_summary":
                results["summary"] = tool_result.get("summary", "")
            elif tool_name == "create_social_media_posts":
                results["social_posts"] = tool_result
            elif tool_name == "create_email_newsletter":
                results["email"] = tool_result
            
            # Add the tool result to the conversation
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(tool_result) if isinstance(tool_result, dict) else tool_result
            })
    
    # If we reach here, the agent couldn't complete the workflow
    print("The agent couldn't complete the workflow within the maximum number of iterations.")
    if results:
        return results
    return {"error": "The agent couldn't complete the workflow within the maximum number of iterations."}

def compare_workflows(blog_post):
    """
    Compare the results of different workflow approaches.
    
    Args:
        blog_post: Dictionary containing the blog post
    
    Returns:
        Dictionary with comparison results
    """
    print("\n=== COMPARING WORKFLOW APPROACHES ===\n")
    
    # Run each workflow
    print("\n--- Running Pipeline Workflow ---")
    pipeline_results = run_pipeline_workflow(blog_post)
    
    print("\n--- Running DAG Workflow with Reflexion ---")
    reflexion_results = run_workflow_with_reflexion(blog_post)
    
    print("\n--- Running Agent-Driven Workflow ---")
    agent_results = run_agent_workflow(blog_post)
    
    # Evaluate each workflow's outputs
    evaluation_results = {}
    workflow_types = {
        "pipeline": pipeline_results,
        "reflexion": reflexion_results, 
        "agent": agent_results
    }
    
    for workflow_name, results in workflow_types.items():
        print(f"\n--- Evaluating {workflow_name} workflow outputs ---")
        
        workflow_evaluation = {}
        
        # Evaluate summary
        if "summary" in results:
            workflow_evaluation["summary"] = evaluate_content(
                results["summary"], 
                "summary"
            )
        
        # Evaluate social media posts
        if "social_posts" in results:
            # Evaluate each platform's post
            platform_evaluations = {}
            for platform, post in results["social_posts"].items():
                platform_evaluations[platform] = evaluate_content(
                    post,
                    f"social_media_post ({platform})"
                )
            workflow_evaluation["social_posts"] = platform_evaluations
        
        # Evaluate email newsletter
        if "email" in results:
            workflow_evaluation["email"] = evaluate_content(
                results["email"].get("body", ""),
                "email"
            )
        
        evaluation_results[workflow_name] = workflow_evaluation
    
    # Generate overall comparison
    comparison = compare_evaluation_results(evaluation_results)
    
    return {
        "pipeline_results": pipeline_results,
        "reflexion_results": reflexion_results,
        "agent_results": agent_results,
        "evaluation_results": evaluation_results,
        "comparison": comparison
    }

def compare_evaluation_results(evaluation_results):
    """
    Compare the evaluation results across different workflows.
    
    Args:
        evaluation_results: Dictionary with evaluation results for each workflow
    
    Returns:
        Dictionary with comparison analysis
    """
    # Calculate average scores for each workflow
    workflow_scores = {}
    for workflow_name, evaluations in evaluation_results.items():
        scores = []
        
        # Get summary score
        if "summary" in evaluations:
            scores.append(evaluations["summary"]["overall_score"])
        
        # Get social media post scores
        if "social_posts" in evaluations:
            for platform, evaluation in evaluations["social_posts"].items():
                scores.append(evaluation["overall_score"])
        
        # Get email score
        if "email" in evaluations:
            scores.append(evaluations["email"]["overall_score"])
        
        # Calculate average score
        workflow_scores[workflow_name] = sum(scores) / len(scores) if scores else 0
    
    # Find the best workflow
    best_workflow = max(workflow_scores.items(), key=lambda x: x[1])[0]
    
    # Prepare comparison analysis
    analysis = {
        "workflow_scores": workflow_scores,
        "best_workflow": best_workflow,
        "analysis": f"The {best_workflow} workflow performed the best overall with a score of {workflow_scores[best_workflow]:.2f}.",
        "strengths_and_weaknesses": {
            "pipeline": {
                "strengths": [
                    "Simple and straightforward implementation",
                    "Predictable execution flow",
                    "Easy to debug and maintain"
                ],
                "weaknesses": [
                    "Limited flexibility in task ordering",
                    "No ability to use results from multiple upstream tasks",
                    "No self-correction mechanism"
                ]
            },
            "reflexion": {
                "strengths": [
                    "Self-correction improves output quality",
                    "More flexible than pipeline workflow",
                    "Explicit evaluation criteria for each task"
                ],
                "weaknesses": [
                    "More complex implementation",
                    "Potentially higher cost due to multiple LLM calls",
                    "May not always converge to better results"
                ]
            },
            "agent": {
                "strengths": [
                    "Most flexible approach",
                    "Can dynamically decide task order",
                    "Ability to handle complex dependencies"
                ],
                "weaknesses": [
                    "Most complex implementation",
                    "Less predictable behavior",
                    "Potential for inefficient tool usage"
                ]
            }
        }
    }
    
    return analysis

def main():
    """
    Main function to run the LLM workflow system.
    """
    # Load the sample blog post
    blog_post = get_sample_blog_post()
    if not blog_post:
        print("Failed to load sample blog post. Exiting.")
        return
    
    print("=== LLM WORKFLOW SYSTEM ===")
    print(f"Using model server: {model_server}")
    print(f"Using model: {LLM_MODEL}")
    
    # Ask the user which workflow to run
    print("\nAvailable workflows:")
    print("1. Pipeline Workflow")
    print("2. DAG Workflow")
    print("3. Chain-of-Thought Extraction")
    print("4. Workflow with Reflexion")
    print("5. Agent-Driven Workflow")
    print("6. Compare All Workflows (will take longer to run)")
    
    choice = input("\nEnter your choice (1-6): ")
    
    if choice == "1":
        results = run_pipeline_workflow(blog_post)
        print("\n=== PIPELINE WORKFLOW RESULTS ===")
        print_results(results)
    elif choice == "2":
        results = run_dag_workflow(blog_post)
        print("\n=== DAG WORKFLOW RESULTS ===")
        print_results(results)
    elif choice == "3":
        key_points = extract_key_points_with_cot(blog_post)
        print("\n=== CHAIN-OF-THOUGHT EXTRACTION RESULTS ===")
        print("Key Points:")
        for i, point in enumerate(key_points, 1):
            print(f"{i}. {point}")
    elif choice == "4":
        results = run_workflow_with_reflexion(blog_post)
        print("\n=== REFLEXION WORKFLOW RESULTS ===")
        print_results(results)
    elif choice == "5":
        results = run_agent_workflow(blog_post)
        print("\n=== AGENT-DRIVEN WORKFLOW RESULTS ===")
        print_results(results)
    elif choice == "6":
        comparison = compare_workflows(blog_post)
        print("\n=== WORKFLOW COMPARISON RESULTS ===")
        print_comparison(comparison)
    else:
        print("Invalid choice. Exiting.")

def print_results(results):
    """
    Print the results of a workflow in a readable format.
    
    Args:
        results: Dictionary with workflow results
    """
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    if "key_points" in results:
        print("\nKey Points:")
        for i, point in enumerate(results["key_points"], 1):
            print(f"{i}. {point}")
    
    if "summary" in results:
        print("\nSummary:")
        print(results["summary"])
    
    if "social_posts" in results:
        print("\nSocial Media Posts:")
        for platform, post in results["social_posts"].items():
            print(f"\n{platform.upper()}:")
            print(post)
    
    if "email" in results:
        print("\nEmail Newsletter:")
        print(f"Subject: {results['email'].get('subject', '')}")
        print(f"Body:\n{results['email'].get('body', '')}")

def print_comparison(comparison):
    """
    Print the comparison results in a readable format.
    
    Args:
        comparison: Dictionary with comparison results
    """
    # Print overall scores
    print("\nOverall Workflow Scores:")
    for workflow, score in comparison["comparison"]["workflow_scores"].items():
        print(f"{workflow.capitalize()} Workflow: {score:.2f}")
    
    # Print best workflow
    best_workflow = comparison["comparison"]["best_workflow"]
    print(f"\nBest Workflow: {best_workflow.capitalize()} (Score: {comparison['comparison']['workflow_scores'][best_workflow]:.2f})")
    
    # Print strengths and weaknesses
    print("\nStrengths and Weaknesses:")
    for workflow, analysis in comparison["comparison"]["strengths_and_weaknesses"].items():
        print(f"\n{workflow.capitalize()} Workflow:")
        
        print("  Strengths:")
        for strength in analysis["strengths"]:
            print(f"  - {strength}")
        
        print("  Weaknesses:")
        for weakness in analysis["weaknesses"]:
            print(f"  - {weakness}")

if __name__ == "__main__":
    main()