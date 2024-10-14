from transformers import pipeline

# Initialize summarization pipeline
summarizer = pipeline("summarization")

# Function to summarize text
def summarize_text(text):
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# Main function
if __name__ == "__main__":
    user_input = input("Enter a paragraph of text to summarize: ")
    summary = summarize_text(user_input)
    print("\nSummary:\n", summary)
    
    
    
