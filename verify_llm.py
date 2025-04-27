from langchain_groq import ChatGroq
import groq

class VerifyWithLLM:
    def __init__(self,llm_api_key):
        self.llm_api_key=llm_api_key
     
    def _verify_author_with_llm(self, candidate_author, context):
        """Use LLM to verify if a person mention is likely an author/contributor"""
        prompt = f"""
        In the following text extract, is "{candidate_author}" likely to be the author or contributor 
        of the content? Answer only YES or NO.
        
        Text: {context}
        """
        
        try:

            client = groq.Client(api_key=self.llm_api_key)
            response = client.chat.completions.create(
                model="llama3-70b-8192",  
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )

            answer = response.choices[0].message.content.strip().upper()
            
            if "YES" in answer:
                return candidate_author
            return None
        except Exception as e:
            print(f"Error verifying author: {e}")
            return candidate_author  
    