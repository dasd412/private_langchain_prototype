from follow_up.InterviewQuestionAnalyzer import InterviewQuestionAnalyzer
from follow_up.FollowUpQuestionGenerator import FollowUpQuestionGenerator
from flow.FlowManager import FlowManager

if __name__ == '__main__':
    # 초기 질문 생성은 아직 안 만들었음.
    # 그래서 나중에 자소서 분석 내용을 토대로 초기질문을 만들어야 함.
    # 분석 내용 역시 아래 코드에도 컨텍스트로 넣어야 함.(메모리 없이 프롬프트 인자로 넣어도 될듯)
    analyzer = InterviewQuestionAnalyzer()
    generator = FollowUpQuestionGenerator()
    flow_manager = FlowManager(analyzer, generator)
    flow_manager.start_flow()




# ['성격의 장점을 말해주세요.', '이전에 출제한 질문이므로 이 질문을 다시 출제하지 않습니다. 새로운 follow up question을 출제하겠습니다.\n\n새로운 follow up question:\n일을 마감안에 끝낼 수 있는 계획적인 성격이라고 말씀하셨는데, 어떤 방식으로 일을 계획하고 마감에 성공하는지 구체적으로 설명해주실 수 있을까요?', '새로운 follow up question:\n구글 플래너와 jira를 사용하여 일정을 관리하고 마감을 성공적으로 이뤄내는 방식에 대해 좀 더 구체적으로 예시를 들어 설명해주실 수 있을까요? 어떤 요소들을 고려하고 어떤 순서로 일을 계획하고 마감을 달성하시는지 알려주세요.']
#
# ('성격의 장점을 말해주세요.', '계획적인 성격으로, 일의 미룸없이 마감안에 일을 끝낼 수 있습니다.', '면접 지원자의 답변에 대해서 좋은 점은 계획적인 성격을 강조하고 일을 미루지 않고 마감안에 끝낼 수 있다는 점을 언급한 것입니다. 이는 회사에서 업무를 효율적으로 수행할 수 있는 능력을 갖추고 있다는 것을 보여줍니다.\n\n하지만 아쉬운 점은 면접 지원자가 성격의 장점을 말하라는 질문에 대해 단 한 가지만 언급한 것입니다. 면접 지원자는 다양한 성격적 장점을 갖고 있을 것으로 예상되는데, 이를 한 가지만 언급하면서 다른 장점들을 감추고 있다는 인상을 줄 수 있습니다. 면접 지원자는 다양한 성격적 장점을 예시와 함께 언급하여 자신의 다양한 장점을 보여줄 수 있었을 것입니다.')
# ('이전에 출제한 질문이므로 이 질문을 다시 출제하지 않습니다. 새로운 follow up question을 출제하겠습니다.\n\n새로운 follow up question:\n일을 마감안에 끝낼 수 있는 계획적인 성격이라고 말씀하셨는데, 어떤 방식으로 일을 계획하고 마감에 성공하는지 구체적으로 설명해주실 수 있을까요?', '구글 플래너를 이용해 일을 계획합니다. 일의 마감은 jira를 활용하고 있습니다.', '면접 지원자의 답변에 대한 좋은 점은 구체적인 도구와 방법을 언급하여 자신의 일정 관리 능력을 보여준다는 것입니다. 구글 플래너와 jira는 일정 관리에 많이 사용되는 도구이므로 면접관에게 익숙한 도구들이라고 생각할 수 있습니다. 이러한 도구를 사용하여 일을 계획하고 마감일을 관리하는 것은 조직적이고 체계적인 업무 처리 방식을 가지고 있다는 것을 나타낼 수 있습니다.\n\n하지만 아쉬운 점은 면접 지원자가 구글 플래너와 jira를 어떻게 활용하는지에 대한 구체적인 설명이 없다는 것입니다. 면접관은 면접 지원자가 어떻게 이 도구들을 활용하여 일정을 관리하고 마감일을 지키는지에 대해 더 자세한 정보를 원할 수 있습니다. 또한, 면접 지원자가 어떻게 이 도구들을 활용하여 효과적으로 일을 계획하고 조직하는지에 대한 예시나 경험을 제시하는 것도 도움이 될 수 있습니다. 이러한 구체적인 설명과 예시를 통해 면접 지원자의 일정 관리 능력을 더욱 강조할 수 있을 것입니다.')
# ('새로운 follow up question:\n구글 플래너와 jira를 사용하여 일정을 관리하고 마감을 성공적으로 이뤄내는 방식에 대해 좀 더 구체적으로 예시를 들어 설명해주실 수 있을까요? 어떤 요소들을 고려하고 어떤 순서로 일을 계획하고 마감을 달성하시는지 알려주세요.', '먼저 모든 일정을 jira에 넣습니다. 그리고 일정의 우선순위를 결정합니다. 우선 순위에서 중요하지 않은 것은 미룹니다. 중요한 일정은 구글 플래너에 기록합니다.', '면접 지원자의 답변에 대한 좋은 점은 다음과 같습니다:\n\n1. 체계적인 접근 방식: 면접 지원자는 모든 일정을 Jira에 넣고, 우선순위를 결정하며 중요하지 않은 일정은 미루는 등 체계적인 접근 방식을 갖고 있다는 것을 보여줍니다. 이는 업무를 효율적으로 관리하고 조직화하는 데 도움이 될 수 있습니다.\n\n2. 다양한 도구 활용: 면접 지원자는 Jira와 구글 플래너와 같은 다양한 도구를 활용하여 업무를 관리하는 것을 언급했습니다. 이는 다양한 도구를 활용하여 업무를 효과적으로 추적하고 조직화하는 능력을 갖고 있다는 것을 보여줍니다.\n\n면접 지원자의 답변에 대한 아쉬운 점은 다음과 같습니다:\n\n1. 구체적인 예시 부족: 면접 지원자는 Jira에 일정을 넣고, 중요한 일정은 구글 플래너에 기록한다고 언급했지만, 이에 대한 구체적인 예시를 제시하지 않았습니다. 예를 들어, 어떤 종류의 일정을 Jira에 넣는지, 어떤 기준으로 우선순위를 결정하는지 등에 대한 구체적인 설명이 부족합니다.\n\n2. 역할과 관련된 내용 부족: 면접 질문은 면접 지원자가 어떻게 업무를 관리하는지에 대한 것이었지만, 면접 지원자는 자신의 역할과 관련된 내용을 언급하지 않았습니다. 예를 들어, 팀의 리더로서 업무를 조정하거나 다른 팀원들과 협력하여 일정을 관리하는 등의 내용을 언급할 수 있었을 것입니다.\n\n3. 결과와 성과에 대한 언급 부족: 면접 지원자는 업무를 어떻게 관리하는지에 대한 접근 방식을 설명했지만, 이에 대한 결과와 성과에 대한 언급이 부족합니다. 예를 들어, 이러한 접근 방식으로 업무를 관리하여 프로젝트의 진행 상황을 향상시키거나 효율성을 향상시킨 경험 등을 언급할 수 있었을 것입니다.')
