from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain  # 라우터 체인에서 default 체인
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # 리스폰스 스트리밍
from langchain.prompts.pipeline import PipelinePromptTemplate  # 프롬프트 합성
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser  # 라우터 체인
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE  # 라우터 체인
from langchain.chains.router import MultiPromptChain  # 라우터 체인


class InterviewQuestionAnalyzer:
    def __init__(self):
        pipeline_prompt = self.__make_prompt_for_analysis()  # 면접 지원자의 면접 답변 분석
        prompt_info = self.__make_more_specific_prompt_with_knowledge(pipeline_prompt)  # "평가 기준에 따라" 면접 지원자의 면접 답변 분석
        self.chain = self.__make_router_chain(pipeline_prompt, prompt_info)  # 만든 프롬프트로 라우터 체인 생성

    def start_analysis(self, question: str, answer: str) -> str:
        analysis = self.chain.run(self.__concat_string(question, answer))

        return analysis

    # private method
    def __concat_string(self, question: str, answer: str) -> str:
        str_list = ["면접 질문 : ", question, "면접 답변 : ", answer]
        return "".join(str_list)

    # private method
    def __make_prompt_for_analysis(self) -> PipelinePromptTemplate:
        full_template = """
        {role}

        {analysis}
        """

        full_prompt = PromptTemplate.from_template(full_template)

        role_template = """
        너는 면접관 경험이 풍부한 면접관이다.
        """

        role_prompt = PromptTemplate.from_template(role_template)

        analysis_template = """
        다음 면접 질문과 면접 지원자의 답변을 보고, 면접 지원자의 답변에 대해서 좋은 점과 아쉬운 점을 분석하라.
        
        면접 질문과 면접 지원자의 답변 :
        {input}

        """

        analysis_prompt = PromptTemplate(template=analysis_template,
                                         input_variables=["input"])

        input_prompts = [
            ("role", role_prompt),
            ("analysis", analysis_prompt)
        ]

        return PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

    # private method
    def __make_more_specific_prompt_with_knowledge(self, pipeline_prompt: PipelinePromptTemplate) -> list[dict]:
        # dict (review_standard_knowledge,review_standard_detail) 형태로 묶어서 만들기
        fit_feature_dict = {
            "job_fit": ("직무 적합성", "직무를 수행하는 데 필요한 기술,지식, 그리고 경험을 가지고 있는지 평가하는 것"),
            "cultural_fit": ("문화 적합성", "조직의 가치와 문화에 잘 맞는지 평가하는 것"),
            "project_management": ("프로젝트 관리 능력", "특정 프로젝트를 기획하고, 이를 성공적으로 실행하고, 필요한 변경 사항을 관리하는지 평가하는 것"),
            "communication": ("의사소통 능력", "자신의 아이디어를 명확하게 전달하고, 다른 사람들과 효과적으로 협업할 수 있는지 평가하는 것"),
            "personality": ("인성 및 태도", "성격, 성실성, 성장 마인드셋을 평가하는 것"),
            "motivation": ("열정 및 지원동기", "왜 그 직무를 선택하고, 그 회사에서 일하길 원하는지 평가하는 것"),
            "adaptability": ("적응력", "새로운 환경이나 상황에 얼마나 빠르게 적응하는지를 평가하는 것"),
            "learning_ability": ("학습 능력", "지식이나 기술을 빠르게 습득하고 새로운 정보를 효과적으로 사용하는 지 평가하는 것"),
            "leadership": ("리더십", "팀에서 리더로서 역할을 수행한 경험이나 리더십에 대한 지식을 평가하는 것")
        }

        knowledge_template = """
        지원자가 {review_standard_detail} 을 {review_standard_knowledge}이라 한다.
        너는 {review_standard_knowledge}의 관점에서 면접 답변을 평가하는데 능숙하다.
        """

        knowledge_prompt = PromptTemplate.from_template(knowledge_template)

        # 분석 체인은 라우팅 체인이므로 라우터 적용.
        prompt_info_array = []

        for fit_feature in fit_feature_dict:
            prompt_info = {
                "name": fit_feature,
                "description": knowledge_prompt.format(
                    review_standard_knowledge=fit_feature_dict[fit_feature][0],
                    review_standard_detail=fit_feature_dict[fit_feature][1],
                ),
                "prompt_template": pipeline_prompt.format(
                    input=""
                ),
            }

            prompt_info_array.append(prompt_info)

        return prompt_info_array

    def __make_router_chain(self, pipeline_prompt: PipelinePromptTemplate,
                            prompt_infos: list[dict]) -> MultiPromptChain:
        llm_router = ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo', verbose=True,
                                streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

        destination_chains = {}

        for prompt_info in prompt_infos:
            name = prompt_info["name"]
            prompt = pipeline_prompt  # <- 프롬프트 합성되어 있음.
            chain = LLMChain(llm=llm_router, prompt=prompt)
            destination_chains[name] = chain

        default_chain = ConversationChain(llm=llm_router, output_key="text")

        destinations = [f"{prompt['name']}:{prompt['description']}" for prompt in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)

        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser())

        router_chain = LLMRouterChain.from_llm(llm_router, router_prompt)

        return MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,
            verbose=True, )
