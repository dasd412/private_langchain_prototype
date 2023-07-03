from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # 리스폰스 스트리밍


class FollowUpQuestionGenerator:
    def __init__(self):
        prompt = self.__make_prompt_for_follow_up()
        self.chain = self.__make_chain_for_follow_up(prompt)

    def generate_follow_up(self, question: str, answer: str, analysis: str, previous_question: list[str]) -> str:
        follow_up = self.chain.predict(question=question, answer=answer, analysis=analysis,
                                       previous_question=''.join(previous_question))
        return follow_up

    # private method
    def __make_prompt_for_follow_up(self) -> PromptTemplate:
        follow_up_template = """
        다음 면접 질문과 답변을 보고, 분석한 내용을 참고 해서 follow up question을 출제 하라. 
        
        동일한 면접 질문과 이전에 출제한 질문은 출제 하지 말고, 대신 새로운 질문을 출제 하라. 

        면접 질문 :
        {question}

        면접 답변:
        {answer}
        
        분석 내용:
        {analysis}

        이전에 출제한 질문:
        {previous_question}
        """

        return PromptTemplate(template=follow_up_template,
                              input_variables=["question", "answer", "analysis", "previous_question"])

    # private method
    def __make_chain_for_follow_up(self, prompt: PromptTemplate) -> LLMChain:
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', verbose=True,
                         streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

        return LLMChain(llm=llm, prompt=prompt)
