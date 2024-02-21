# Latest Large Language Model Attack Papers
**update at 2024-02-21 10:58:20**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Prompt Stealing Attacks Against Large Language Models**

针对大型语言模型的即时窃取攻击 cs.CR

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.12959v1) [paper-pdf](http://arxiv.org/pdf/2402.12959v1)

**Authors**: Zeyang Sha, Yang Zhang

**Abstract**: The increasing reliance on large language models (LLMs) such as ChatGPT in various fields emphasizes the importance of ``prompt engineering,'' a technology to improve the quality of model outputs. With companies investing significantly in expert prompt engineers and educational resources rising to meet market demand, designing high-quality prompts has become an intriguing challenge. In this paper, we propose a novel attack against LLMs, named prompt stealing attacks. Our proposed prompt stealing attack aims to steal these well-designed prompts based on the generated answers. The prompt stealing attack contains two primary modules: the parameter extractor and the prompt reconstruction. The goal of the parameter extractor is to figure out the properties of the original prompts. We first observe that most prompts fall into one of three categories: direct prompt, role-based prompt, and in-context prompt. Our parameter extractor first tries to distinguish the type of prompts based on the generated answers. Then, it can further predict which role or how many contexts are used based on the types of prompts. Following the parameter extractor, the prompt reconstructor can be used to reconstruct the original prompts based on the generated answers and the extracted features. The final goal of the prompt reconstructor is to generate the reversed prompts, which are similar to the original prompts. Our experimental results show the remarkable performance of our proposed attacks. Our proposed attacks add a new dimension to the study of prompt engineering and call for more attention to the security issues on LLMs.

摘要: 各个领域对ChatGPT等大型语言模型(LLM)的日益依赖，突出了“快速工程”的重要性，这是一种提高模型输出质量的技术。随着公司在专家提示工程师和教育资源上的大量投资以满足市场需求，设计高质量的提示已成为一项耐人寻味的挑战。在本文中，我们提出了一种新的针对LLMS的攻击，称为即时窃取攻击。我们提出的提示窃取攻击旨在根据生成的答案窃取这些精心设计的提示。即时窃取攻击包括两个主要模块：参数提取和即时重构。参数提取程序的目标是找出原始提示的属性。我们首先观察到，大多数提示可以分为三类：直接提示、基于角色的提示和上下文提示。我们的参数提取程序首先尝试根据生成的答案区分提示的类型。然后，它可以根据提示的类型进一步预测使用了哪个角色或多少个上下文。在参数抽取器之后，提示重建器可用于基于生成的答案和提取的特征来重建原始提示。提示重建器的最终目标是生成与原始提示类似的反向提示。我们的实验结果表明，我们提出的攻击具有显著的性能。我们提出的攻击为即时工程的研究增加了一个新的维度，并呼吁更多地关注LLMS上的安全问题。



## **2. Measuring Impacts of Poisoning on Model Parameters and Neuron Activations: A Case Study of Poisoning CodeBERT**

测量中毒对模型参数和神经元激活的影响：中毒CodeBERT的案例研究 cs.SE

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.12936v1) [paper-pdf](http://arxiv.org/pdf/2402.12936v1)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Navid Ayoobi, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have revolutionized software development practices, yet concerns about their safety have arisen, particularly regarding hidden backdoors, aka trojans. Backdoor attacks involve the insertion of triggers into training data, allowing attackers to manipulate the behavior of the model maliciously. In this paper, we focus on analyzing the model parameters to detect potential backdoor signals in code models. Specifically, we examine attention weights and biases, activation values, and context embeddings of the clean and poisoned CodeBERT models. Our results suggest noticeable patterns in activation values and context embeddings of poisoned samples for the poisoned CodeBERT model; however, attention weights and biases do not show any significant differences. This work contributes to ongoing efforts in white-box detection of backdoor signals in LLMs of code through the analysis of parameters and activations.

摘要: 大型语言模型(LLM)使软件开发实践发生了革命性的变化，但也出现了对其安全性的担忧，特别是关于隐藏的后门，也就是特洛伊木马。后门攻击包括在训练数据中插入触发器，允许攻击者恶意操纵模型的行为。在本文中，我们重点分析模型参数，以检测代码模型中潜在的后门信号。具体地说，我们检查了干净的和有毒的CodeBERT模型的注意力权重和偏差、激活值和上下文嵌入。我们的结果表明，对于中毒的CodeBERT模型，中毒样本的激活值和上下文嵌入有明显的模式；然而，注意力权重和偏差没有显示出任何显著的差异。这项工作有助于通过分析参数和激活来对代码的LLMS中的后门信号进行白盒检测。



## **3. Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!**

模拟失调：大型语言模型的安全校准可能会适得其反！ cs.CL

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12343v1) [paper-pdf](http://arxiv.org/pdf/2402.12343v1)

**Authors**: Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao

**Abstract**: Large language models (LLMs) need to undergo safety alignment to ensure safe conversations with humans. However, in this work, we introduce an inference-time attack framework, demonstrating that safety alignment can also unintentionally facilitate harmful outcomes under adversarial manipulation. This framework, named Emulated Disalignment (ED), adversely combines a pair of open-source pre-trained and safety-aligned language models in the output space to produce a harmful language model without any training. Our experiments with ED across three datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show that ED doubles the harmfulness of pre-trained models and outperforms strong baselines, achieving the highest harmful rate in 43 out of 48 evaluation subsets by a large margin. Crucially, our findings highlight the importance of reevaluating the practice of open-sourcing language models even after safety alignment.

摘要: 大型语言模型(LLM)需要经过安全调整，以确保与人类的安全对话。然而，在这项工作中，我们引入了一个推理时间攻击框架，证明了安全对齐也可以在无意中促进对抗性操纵下的有害结果。这个名为仿真失调(ED)的框架在输出空间中反向组合了两个开放源码的预训练和安全对齐的语言模型，在没有任何训练的情况下产生了有害的语言模型。我们在三个数据集和四个模型家族(骆驼-1、骆驼-2、米斯特拉尔和羊驼)上使用ED进行的实验表明，ED的危害性是预训练模型的两倍，并且性能优于强基线，在48个评估子集中的43个子集上获得了最高的伤害率。至关重要的是，我们的发现强调了重新评估开源语言模型实践的重要性，即使在安全调整之后也是如此。



## **4. Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models**

Robust CLIP：用于强健大型视觉语言模型的视觉嵌入的无监督对抗性微调 cs.LG

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12336v1) [paper-pdf](http://arxiv.org/pdf/2402.12336v1)

**Authors**: Christian Schlarmann, Naman Deep Singh, Francesco Croce, Matthias Hein

**Abstract**: Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many vision-language models (VLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (VLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of VLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP model with our robust one. No retraining or fine-tuning of the VLM is required. The code and robust models are available at https://github.com/chs20/RobustVLM

摘要: OpenFlamingo、LLaVA和GPT-4等多模式基础模型越来越多地用于各种实际任务。先前的工作表明，这些模型非常容易受到视觉通道的对抗性攻击。这些攻击可以被用来传播虚假信息或欺骗用户，从而构成巨大的风险，这使得大型多通道基础模型的健壮性成为一个紧迫的问题。在许多视觉语言模型(VLM)中，剪辑模型或其变体之一被用作冻结的视觉编码器，例如LLaVA和OpenFlamingo。我们提出了一种无监督的对抗性微调方案，以获得一个健壮的裁剪视觉编码器，它对依赖于裁剪的所有视觉下游任务(VLM，零镜头分类)都具有健壮性。特别是，我们表明，一旦用我们的健壮模型取代了原始的剪辑模型，恶意第三方提供的篡改图像就不再可能对VLM的用户进行秘密攻击。不需要对VLM进行再培训或微调。代码和健壮模型可在https://github.com/chs20/RobustVLM上获得



## **5. Polarization of Autonomous Generative AI Agents Under Echo Chambers**

回声腔下自主生成性人工智能智能体的极化 cs.CL

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12212v1) [paper-pdf](http://arxiv.org/pdf/2402.12212v1)

**Authors**: Masaya Ohagi

**Abstract**: Online social networks often create echo chambers where people only hear opinions reinforcing their beliefs. An echo chamber often generates polarization, leading to conflicts caused by people with radical opinions, such as the January 6, 2021, attack on the US Capitol. The echo chamber has been viewed as a human-specific problem, but this implicit assumption is becoming less reasonable as large language models, such as ChatGPT, acquire social abilities. In response to this situation, we investigated the potential for polarization to occur among a group of autonomous AI agents based on generative language models in an echo chamber environment. We had AI agents discuss specific topics and analyzed how the group's opinions changed as the discussion progressed. As a result, we found that the group of agents based on ChatGPT tended to become polarized in echo chamber environments. The analysis of opinion transitions shows that this result is caused by ChatGPT's high prompt understanding ability to update its opinion by considering its own and surrounding agents' opinions. We conducted additional experiments to investigate under what specific conditions AI agents tended to polarize. As a result, we identified factors that strongly influence polarization, such as the agent's persona. These factors should be monitored to prevent the polarization of AI agents.

摘要: 在线社交网络通常会产生回音室，人们只会听到强化自己信念的意见。回音室往往会产生两极分化，导致有激进观点的人引发冲突，比如2021年1月6日对美国国会大厦的袭击。回音室一直被视为人类特有的问题，但随着ChatGPT等大型语言模型获得社交能力，这种隐含的假设变得越来越不合理。针对这种情况，我们研究了在回声室环境中基于生成语言模型的一组自主AI代理之间发生极化的可能性。我们让人工智能代理讨论特定的主题，并分析随着讨论的进行，小组的意见如何变化。结果，我们发现基于ChatGPT的代理组在回声室环境中倾向于极化。对意见转换的分析表明，这一结果是由ChatGPT的高及时理解能力所导致的，该能力通过考虑自身和周围代理的意见来更新其意见。我们进行了额外的实验来研究AI代理在什么特定条件下倾向于失败。因此，我们确定了强烈影响极化的因素，例如代理人的角色。应该监控这些因素，以防止AI代理的极化。



## **6. On the Safety Concerns of Deploying LLMs/VLMs in Robotics: Highlighting the Risks and Vulnerabilities**

在机器人中部署LLMS/VLM的安全问题：突出风险和漏洞 cs.RO

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.10340v2) [paper-pdf](http://arxiv.org/pdf/2402.10340v2)

**Authors**: Xiyang Wu, Ruiqi Xian, Tianrui Guan, Jing Liang, Souradip Chakraborty, Fuxiao Liu, Brian Sadler, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works have focused on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation, navigation, etc. However, such integration can introduce significant vulnerabilities, in terms of their susceptibility to adversarial attacks due to the language models, potentially leading to catastrophic consequences. By examining recent works at the interface of LLMs/VLMs and robotics, we show that it is easy to manipulate or misguide the robot's actions, leading to safety hazards. We define and provide examples of several plausible adversarial attacks, and conduct experiments on three prominent robot frameworks integrated with a language model, including KnowNo VIMA, and Instruct2Act, to assess their susceptibility to these attacks. Our empirical findings reveal a striking vulnerability of LLM/VLM-robot integrated systems: simple adversarial attacks can significantly undermine the effectiveness of LLM/VLM-robot integrated systems. Specifically, our data demonstrate an average performance deterioration of 21.2% under prompt attacks and a more alarming 30.2% under perception attacks. These results underscore the critical need for robust countermeasures to ensure the safe and reliable deployment of the advanced LLM/VLM-based robotic systems.

摘要: 在这篇文章中，我们强调了与将大语言模型(LLM)和视觉语言模型(VLM)集成到机器人应用中相关的健壮性和安全性的关键问题。最近的工作集中在使用LLMS和VLM来提高机器人任务的性能，如操纵、导航等。然而，这种集成可能会引入显著的漏洞，因为它们容易由于语言模型而受到对手攻击，可能会导致灾难性的后果。通过对LLMS/VLMS与机器人接口的最新研究，我们发现很容易操纵或误导机器人的动作，从而导致安全隐患。我们定义并提供了几种可能的对抗性攻击的例子，并在三个与语言模型集成的著名机器人框架上进行了实验，包括KnowNo Vima和Instruct2Act，以评估它们对这些攻击的敏感度。我们的实验结果揭示了LLM/VLM-机器人集成系统的一个显著漏洞：简单的对抗性攻击会显著削弱LLM/VLM-机器人集成系统的有效性。具体地说，我们的数据显示，在即时攻击下，平均性能下降21.2%，在感知攻击下，更令人震惊的是30.2%。这些结果突出表明，迫切需要强有力的对策，以确保安全可靠地部署先进的基于LLM/VLM的机器人系统。



## **7. SPML: A DSL for Defending Language Models Against Prompt Attacks**

SPML：一种用于保护语言模型免受提示攻击的DSL cs.LG

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.11755v1) [paper-pdf](http://arxiv.org/pdf/2402.11755v1)

**Authors**: Reshabh K Sharma, Vinayak Gupta, Dan Grossman

**Abstract**: Large language models (LLMs) have profoundly transformed natural language applications, with a growing reliance on instruction-based definitions for designing chatbots. However, post-deployment the chatbot definitions are fixed and are vulnerable to attacks by malicious users, emphasizing the need to prevent unethical applications and financial losses. Existing studies explore user prompts' impact on LLM-based chatbots, yet practical methods to contain attacks on application-specific chatbots remain unexplored. This paper presents System Prompt Meta Language (SPML), a domain-specific language for refining prompts and monitoring the inputs to the LLM-based chatbots. SPML actively checks attack prompts, ensuring user inputs align with chatbot definitions to prevent malicious execution on the LLM backbone, optimizing costs. It also streamlines chatbot definition crafting with programming language capabilities, overcoming natural language design challenges. Additionally, we introduce a groundbreaking benchmark with 1.8k system prompts and 20k user inputs, offering the inaugural language and benchmark for chatbot definition evaluation. Experiments across datasets demonstrate SPML's proficiency in understanding attacker prompts, surpassing models like GPT-4, GPT-3.5, and LLAMA. Our data and codes are publicly available at: https://prompt-compiler.github.io/SPML/.

摘要: 大型语言模型（LLM）已经深刻地改变了自然语言应用程序，越来越依赖于基于解释的定义来设计聊天机器人。然而，部署后的聊天机器人定义是固定的，容易受到恶意用户的攻击，强调需要防止不道德的应用程序和财务损失。现有的研究探讨了用户提示对基于LLM的聊天机器人的影响，但遏制对特定应用程序聊天机器人的攻击的实用方法尚未探索。本文介绍了系统提示Meta语言（SPML），这是一种特定于领域的语言，用于细化提示和监控基于LLM的聊天机器人的输入。SPML主动检查攻击提示，确保用户输入与聊天机器人定义保持一致，以防止在LLM主干上恶意执行，从而优化成本。它还利用编程语言功能简化了聊天机器人定义，克服了自然语言设计的挑战。此外，我们还引入了一个具有1.8k系统提示和20 k用户输入的开创性基准，为聊天机器人定义评估提供了首个语言和基准。跨数据集的实验证明了SPML在理解攻击者提示方面的熟练程度，超过了GPT-4、GPT-3.5和LLAMA等模型。我们的数据和代码可在https://prompt-compiler.github.io/SPML/上公开获取。



## **8. ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs**

ArtPrompt：基于ASCII ART的针对结盟LLM的越狱攻击 cs.CL

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.11753v1) [paper-pdf](http://arxiv.org/pdf/2402.11753v1)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Zhen Xiang, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran

**Abstract**: Safety is critical to the usage of large language models (LLMs). Multiple techniques such as data filtering and supervised fine-tuning have been developed to strengthen LLM safety. However, currently known techniques presume that corpora used for safety alignment of LLMs are solely interpreted by semantics. This assumption, however, does not hold in real-world applications, which leads to severe vulnerabilities in LLMs. For example, users of forums often use ASCII art, a form of text-based art, to convey image information. In this paper, we propose a novel ASCII art-based jailbreak attack and introduce a comprehensive benchmark Vision-in-Text Challenge (ViTC) to evaluate the capabilities of LLMs in recognizing prompts that cannot be solely interpreted by semantics. We show that five SOTA LLMs (GPT-3.5, GPT-4, Gemini, Claude, and Llama2) struggle to recognize prompts provided in the form of ASCII art. Based on this observation, we develop the jailbreak attack ArtPrompt, which leverages the poor performance of LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs. ArtPrompt only requires black-box access to the victim LLMs, making it a practical attack. We evaluate ArtPrompt on five SOTA LLMs, and show that ArtPrompt can effectively and efficiently induce undesired behaviors from all five LLMs.

摘要: 安全对于大型语言模型(LLM)的使用至关重要。已经开发了多种技术，如数据过滤和有监督的微调，以加强LLM的安全性。然而，目前已知的技术假定用于LLM的安全对准的语料库仅由语义解释。然而，这一假设在现实世界的应用程序中并不成立，这导致了LLMS中的严重漏洞。例如，论坛的用户经常使用ASCII艺术，这是一种基于文本的艺术形式，以传达图像信息。本文提出了一种新的基于ASCII ART的越狱攻击方法，并引入了一个综合基准的文本中视觉挑战(VITC)来评估LLMS在识别不能完全由语义解释的提示方面的能力。我们发现，五个SOTA LLM(GPT-3.5、GPT-4、双子座、克劳德和Llama2)很难识别以ASCII ART形式提供的提示。基于这种观察，我们开发了越狱攻击ArtPrompt，它利用LLMS在识别ASCII ART方面的较差性能来绕过安全措施，并从LLM引发不希望看到的行为。ArtPrompt只需要黑盒访问受攻击的LLM，这使其成为一种实际的攻击。我们在五个SOTA LLM上对ArtPrompt进行了评估，结果表明，ArtPrompt可以有效和高效地诱导所有五个LLM的不良行为。



## **9. Vaccine: Perturbation-aware Alignment for Large Language Model**

疫苗：大型语言模型中的扰动感知比对 cs.LG

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.01109v2) [paper-pdf](http://arxiv.org/pdf/2402.01109v2)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.

摘要: 微调即服务的新范式为大型语言模型（LLM）引入了一个新的攻击面：用户上传的一些有害数据可以很容易地欺骗微调，从而产生一个破坏性的模型。我们进行了实证分析，并揭示了\textit{有害嵌入漂移}现象，显示了一个可能的原因，破坏效果。受我们研究结果的启发，我们提出了Vaccine，一种扰动感知对齐技术，以减轻用户微调的安全风险。Vaccine的核心思想是通过在对齐阶段逐步添加精心制作的扰动来产生不变的隐藏嵌入。这使得嵌入能够在微调阶段承受来自未净化的用户数据的有害扰动。我们对开源主流LLM的研究结果（例如，Llama 2，Opt，Vicuna）证明了Vaccine可以提高比对对有害提示诱导的嵌入漂移的鲁棒性，同时保留对良性提示的推理能力。我们的代码可以在\url{https：//github.com/git-disl/Vaccine}上找到。



## **10. Stumbling Blocks: Stress Testing the Robustness of Machine-Generated Text Detectors Under Attacks**

绊脚石：压力测试机器生成的文本检测器在攻击下的健壮性 cs.CL

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.11638v1) [paper-pdf](http://arxiv.org/pdf/2402.11638v1)

**Authors**: Yichen Wang, Shangbin Feng, Abe Bohan Hou, Xiao Pu, Chao Shen, Xiaoming Liu, Yulia Tsvetkov, Tianxing He

**Abstract**: The widespread use of large language models (LLMs) is increasing the demand for methods that detect machine-generated text to prevent misuse. The goal of our study is to stress test the detectors' robustness to malicious attacks under realistic scenarios. We comprehensively study the robustness of popular machine-generated text detectors under attacks from diverse categories: editing, paraphrasing, prompting, and co-generating. Our attacks assume limited access to the generator LLMs, and we compare the performance of detectors on different attacks under different budget levels. Our experiments reveal that almost none of the existing detectors remain robust under all the attacks, and all detectors exhibit different loopholes. Averaging all detectors, the performance drops by 35% across all attacks. Further, we investigate the reasons behind these defects and propose initial out-of-the-box patches to improve robustness.

摘要: 大型语言模型(LLM)的广泛使用增加了对检测机器生成的文本以防止滥用的方法的需求。我们研究的目标是重点测试检测器在真实场景下对恶意攻击的健壮性。我们全面研究了流行的机器生成文本检测器在不同类别的攻击下的健壮性：编辑、释义、提示和联合生成。我们的攻击假设对生成器LLM的访问是有限的，并比较了不同预算水平下检测器在不同攻击下的性能。我们的实验表明，几乎没有一个现有的检测器在所有攻击下都保持健壮，并且所有的检测器都显示出不同的漏洞。平均所有检测器，在所有攻击中性能下降35%。此外，我们调查了这些缺陷背后的原因，并提出了初始的开箱即用补丁来提高健壮性。



## **11. OUTFOX: LLM-Generated Essay Detection Through In-Context Learning with Adversarially Generated Examples**

Outfox：基于上下文学习的LLM生成的文章检测与恶意生成的示例 cs.CL

AAAI 2024 camera ready. Code and dataset available at  https://github.com/ryuryukke/OUTFOX

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2307.11729v3) [paper-pdf](http://arxiv.org/pdf/2307.11729v3)

**Authors**: Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors lack robustness against attacks: they degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, a malicious user might attempt to deliberately evade the detectors based on detection results, but this has not been assumed in previous studies. In this paper, we propose OUTFOX, a framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output. In this framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect, while the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Experiments in the domain of student essays show that the proposed detector improves the detection performance on the attacker-generated texts by up to +41.3 points F1-score. Furthermore, the proposed detector shows a state-of-the-art detection performance: up to 96.9 points F1-score, beating existing detectors on non-attacked texts. Finally, the proposed attacker drastically degrades the performance of detectors by up to -57.0 points F1-score, massively outperforming the baseline paraphrasing method for evading detection.

摘要: 大型语言模型（LLM）在文本生成方面已经达到了人类水平的流畅性，因此很难区分人类编写的文本和LLM生成的文本。这造成了越来越多的LLM滥用的风险，并要求开发检测器来识别LLM生成的文本。然而，现有的检测器缺乏对攻击的鲁棒性：它们通过简单地解释LLM生成的文本来降低检测精度。此外，恶意用户可能会试图故意逃避检测器的检测结果的基础上，但这在以前的研究中没有假设。在本文中，我们提出了OUTFOX，一个框架，提高了LLM生成的文本检测器的鲁棒性，允许检测器和攻击者考虑对方的输出。在这个框架中，攻击者使用检测器的预测标签作为上下文学习的示例，并对抗性地生成更难检测的文章，而检测器使用对抗性地生成的文章作为上下文学习的示例，以学习检测来自强大攻击者的文章。在学生论文领域的实验表明，该检测器提高了攻击者生成的文本的检测性能高达+41.3点F1分数。此外，所提出的检测器显示出最先进的检测性能：高达96.9点的F1分数，击败现有的检测器对非攻击文本。最后，所提出的攻击者大大降低了检测器的性能高达-57.0点F1分数，大大优于基线释义方法逃避检测。



## **12. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

披着羊皮的狼：广义嵌套越狱提示可以轻松愚弄大型语言模型 cs.CL

Pre-print, code is available at https://github.com/NJUNLP/ReNeLLM

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2311.08268v2) [paper-pdf](http://arxiv.org/pdf/2311.08268v2)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, compromising generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.

摘要: 大型语言模型(LLM)，如ChatGPT和GPT-4，旨在提供有用和安全的响应。然而，被称为“越狱”的对抗性提示可能会绕过安全措施，导致LLMS生成潜在的有害内容。探索越狱提示可以帮助更好地揭示LLM的弱点，并进一步指导我们确保它们的安全。不幸的是，现有的越狱方法要么需要复杂的人工设计，要么需要对其他白盒模型进行优化，从而影响通用性或效率。本文将越狱提示攻击概括为两个方面：(1)提示重写和(2)场景嵌套。在此基础上，我们提出了ReNeLLM，这是一个利用LLM自身生成有效越狱提示的自动化框架。广泛的实验表明，与现有的基准相比，ReNeLLM显著提高了攻击成功率，同时大大降低了时间成本。我们的研究也揭示了现有防御方法在保护低密度脂蛋白方面的不足。最后，从即时执行优先级的角度分析了LLMS防御失败的原因，并提出了相应的防御策略。我们希望我们的研究能够促进学术界和低成本管理系统开发商提供更安全和更规范的低成本管理系统。代码可在https://github.com/NJUNLP/ReNeLLM.上获得



## **13. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

基于困惑度量和上下文信息的令牌级敌意提示检测 cs.CL

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2311.11509v3) [paper-pdf](http://arxiv.org/pdf/2311.11509v3)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that mislead LLMs into generating incorrect or undesired outputs. Previous work has revealed that with relatively simple yet effective attacks based on discrete optimization, it is possible to generate adversarial prompts that bypass moderation and alignment of the models. This vulnerability to adversarial prompts underscores a significant concern regarding the robustness and reliability of LLMs. Our work aims to address this concern by introducing a novel approach to detecting adversarial prompts at a token level, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity, where tokens predicted with high probability are considered normal, and those exhibiting high perplexity are flagged as adversarial. Additionaly, our method also integrates context understanding by incorporating neighboring token information to encourage the detection of contiguous adversarial prompt sequences. To this end, we design two algorithms for adversarial prompt detection: one based on optimization techniques and another on Probabilistic Graphical Models (PGM). Both methods are equipped with efficient solving methods, ensuring efficient adversarial prompt detection. Our token-level detection result can be visualized as heatmap overlays on the text sequence, allowing for a clearer and more intuitive representation of which part of the text may contain adversarial prompts.

摘要: 近年来，大型语言模型(LLM)已经成为各种应用中的关键工具。然而，这些模型容易受到敌意提示攻击，攻击者可以仔细策划误导LLM生成不正确或不想要的输出的输入字符串。以前的工作已经表明，通过基于离散优化的相对简单但有效的攻击，有可能生成绕过模型的缓和和对齐的对抗性提示。这种对敌意提示的脆弱性突出了人们对LLMS的健壮性和可靠性的严重关切。我们的工作旨在通过引入一种新的方法来检测令牌级别的敌意提示，利用LLM预测下一个令牌的概率的能力来解决这一问题。我们测量了模型的困惑程度，其中高概率预测的标记被认为是正常的，而那些表现出高困惑的标记被标记为对抗性的。此外，我们的方法还通过结合邻近的令牌信息来整合上下文理解，以鼓励检测连续的对抗性提示序列。为此，我们设计了两种对抗性提示检测算法：一种基于优化技术，另一种基于概率图模型(PGM)。这两种方法都配备了高效的解决方法，确保了高效的对抗性及时检测。我们的令牌级检测结果可以可视化为覆盖在文本序列上的热图，从而允许更清晰、更直观地表示文本的哪一部分可能包含对抗性提示。



## **14. Effective Prompt Extraction from Language Models**

从语言模型中有效地提取提示 cs.CL

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2307.06865v2) [paper-pdf](http://arxiv.org/pdf/2307.06865v2)

**Authors**: Yiming Zhang, Nicholas Carlini, Daphne Ippolito

**Abstract**: The text generated by large language models is commonly controlled by prompting, where a prompt prepended to a user's query guides the model's output. The prompts used by companies to guide their models are often treated as secrets, to be hidden from the user making the query. They have even been treated as commodities to be bought and sold. However, anecdotal reports have shown adversarial users employing prompt extraction attacks to recover these prompts. In this paper, we present a framework for systematically measuring the effectiveness of these attacks. In experiments with 3 different sources of prompts and 11 underlying large language models, we find that simple text-based attacks can in fact reveal prompts with high probability. Our framework determines with high precision whether an extracted prompt is the actual secret prompt, rather than a model hallucination. Prompt extraction experiments on real systems such as Bing Chat and ChatGPT suggest that system prompts can be revealed by an adversary despite existing defenses in place.

摘要: 大型语言模型生成的文本通常通过提示进行控制，其中用户查询前的提示将指导模型的输出。公司用来指导其模型的提示通常被视为秘密，对进行查询的用户隐藏。它们甚至被视为可以买卖的商品。然而，坊间报道显示，敌意用户使用提示提取攻击来恢复这些提示。在本文中，我们提出了一个系统地衡量这些攻击的有效性的框架。在对3种不同的提示源和11个基本的大型语言模型进行的实验中，我们发现简单的基于文本的攻击实际上可以高概率地揭示提示。我们的框架高精度地确定提取的提示是否是实际的秘密提示，而不是模型幻觉。在Bing Chat和ChatGPT等真实系统上的提示提取实验表明，尽管现有的防御措施已经到位，但系统提示可以被对手泄露。



## **15. Can Large Language Models perform Relation-based Argument Mining?**

大型语言模型可以执行基于关系的参数挖掘吗？ cs.CL

10 pages, 9 figures, submitted to ACL 2024

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2402.11243v1) [paper-pdf](http://arxiv.org/pdf/2402.11243v1)

**Authors**: Deniz Gorur, Antonio Rago, Francesca Toni

**Abstract**: Argument mining (AM) is the process of automatically extracting arguments, their components and/or relations amongst arguments and components from text. As the number of platforms supporting online debate increases, the need for AM becomes ever more urgent, especially in support of downstream tasks. Relation-based AM (RbAM) is a form of AM focusing on identifying agreement (support) and disagreement (attack) relations amongst arguments. RbAM is a challenging classification task, with existing methods failing to perform satisfactorily. In this paper, we show that general-purpose Large Language Models (LLMs), appropriately primed and prompted, can significantly outperform the best performing (RoBERTa-based) baseline. Specifically, we experiment with two open-source LLMs (Llama-2 and Mistral) with ten datasets.

摘要: 参数挖掘是从文本中自动提取参数、它们的组成部分和/或参数和组成部分之间的关系的过程。随着支持在线辩论的平台数量的增加，对AM的需求变得更加迫切，特别是在支持下游任务方面。基于关系的AM(RbAM)是AM的一种形式，专注于识别论点之间的一致(支持)和不一致(攻击)关系。RbAM是一项具有挑战性的分类任务，现有的方法无法令人满意地执行。在这篇文章中，我们证明了通用的大型语言模型(LLM)，经过适当的启动和提示，可以显著超过最佳性能(基于Roberta)的基线。具体地说，我们用两个开放源码的LLMS(Llama-2和Mistral)和10个数据集进行了实验。



## **16. Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents**

当心你的特工！调查对基于LLM的代理的后门威胁 cs.CR

The first two authors contribute equally. Code and data are available  at https://github.com/lancopku/agent-backdoor-attacks

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2402.11208v1) [paper-pdf](http://arxiv.org/pdf/2402.11208v1)

**Authors**: Wenkai Yang, Xiaohan Bi, Yankai Lin, Sishuo Chen, Jie Zhou, Xu Sun

**Abstract**: Leveraging the rapid development of Large Language Models LLMs, LLM-based agents have been developed to handle various real-world applications, including finance, healthcare, and shopping, etc. It is crucial to ensure the reliability and security of LLM-based agents during applications. However, the safety issues of LLM-based agents are currently under-explored. In this work, we take the first step to investigate one of the typical safety threats, backdoor attack, to LLM-based agents. We first formulate a general framework of agent backdoor attacks, then we present a thorough analysis on the different forms of agent backdoor attacks. Specifically, from the perspective of the final attacking outcomes, the attacker can either choose to manipulate the final output distribution, or only introduce malicious behavior in the intermediate reasoning process, while keeping the final output correct. Furthermore, the former category can be divided into two subcategories based on trigger locations: the backdoor trigger can be hidden either in the user query or in an intermediate observation returned by the external environment. We propose the corresponding data poisoning mechanisms to implement the above variations of agent backdoor attacks on two typical agent tasks, web shopping and tool utilization. Extensive experiments show that LLM-based agents suffer severely from backdoor attacks, indicating an urgent need for further research on the development of defenses against backdoor attacks on LLM-based agents. Warning: This paper may contain biased content.

摘要: 利用大型语言模型LLM的快速发展，基于LLM的代理已经被开发出来处理各种现实世界的应用，包括金融、医疗保健和购物等。在应用过程中，确保基于LLM的代理的可靠性和安全性至关重要。然而，基于LLM的制剂的安全性问题目前还没有得到充分的研究。在这项工作中，我们首先调查了LLM代理面临的一种典型的安全威胁--后门攻击。我们首先建立了代理后门攻击的一般框架，然后对代理后门攻击的不同形式进行了深入的分析。具体地说，从最终攻击结果的角度来看，攻击者可以选择操纵最终输出分布，也可以只在中间推理过程中引入恶意行为，同时保持最终输出的正确性。此外，根据触发器的位置，前一类可以分为两个子类别：后门触发器可以隐藏在用户查询中，也可以隐藏在外部环境返回的中间观察中。针对网络购物和工具利用这两种典型的代理任务，我们提出了相应的数据毒化机制来实现上述代理后门攻击的变体。大量的实验表明，基于LLM的代理遭受严重的后门攻击，这表明迫切需要进一步研究开发针对基于LLM的代理的后门攻击的防御措施。警告：此论文可能包含有偏见的内容。



## **17. Open the Pandora's Box of LLMs: Jailbreaking LLMs through Representation Engineering**

打开LLMS的潘多拉盒子：通过表征工程越狱LLMS cs.CL

13 pages, 9 figures

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2401.06824v2) [paper-pdf](http://arxiv.org/pdf/2401.06824v2)

**Authors**: Tianlong Li, Shihan Dou, Wenhao Liu, Muling Wu, Changze Lv, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: Jailbreaking techniques aim to probe the boundaries of safety in large language models (LLMs) by inducing them to generate toxic responses to malicious queries, a significant concern within the LLM community. While existing jailbreaking methods primarily rely on prompt engineering, altering inputs to evade LLM safety mechanisms, they suffer from low attack success rates and significant time overheads, rendering them inflexible. To overcome these limitations, we propose a novel jailbreaking approach, named Jailbreaking LLMs through Representation Engineering (JRE). Our method requires only a small number of query pairs to extract ``safety patterns'' that can be used to circumvent the target model's defenses, achieving unprecedented jailbreaking performance. Building upon these findings, we also introduce a novel defense framework inspired by JRE principles, which demonstrates notable effectiveness. Extensive experimentation confirms the superior performance of the JRE attacks and the robustness of the JRE defense framework. We hope this study contributes to advancing the understanding of model safety issues through the lens of representation engineering.

摘要: 越狱技术旨在通过诱导大型语言模型(LLM)对恶意查询产生有毒响应来探测它们的安全边界，这是LLM社区内的一个重大问题。虽然现有的越狱方法主要依赖于即时工程，改变输入来规避LLM安全机制，但它们存在攻击成功率低和大量时间开销的问题，使其缺乏灵活性。为了克服这些局限性，我们提出了一种新的越狱方法，称为通过表示工程的越狱LLMS(JRE)。我们的方法只需要少量的查询对来提取可以用来绕过目标模型的防御的“安全模式”，从而获得前所未有的越狱性能。在这些发现的基础上，我们还引入了一个新的防御框架，该框架受到JRE原则的启发，表现出显著的有效性。大量实验证实了JRE攻击的优越性能和JRE防御框架的健壮性。我们希望这项研究有助于通过表征工程的视角来促进对模型安全问题的理解。



## **18. VQAttack: Transferable Adversarial Attacks on Visual Question Answering via Pre-trained Models**

VQAttack：基于预训练模型的可转移敌意视觉问答攻击 cs.CV

AAAI 2024, 11 pages

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.11083v1) [paper-pdf](http://arxiv.org/pdf/2402.11083v1)

**Authors**: Ziyi Yin, Muchao Ye, Tianrong Zhang, Jiaqi Wang, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma

**Abstract**: Visual Question Answering (VQA) is a fundamental task in computer vision and natural language process fields. Although the ``pre-training & finetuning'' learning paradigm significantly improves the VQA performance, the adversarial robustness of such a learning paradigm has not been explored. In this paper, we delve into a new problem: using a pre-trained multimodal source model to create adversarial image-text pairs and then transferring them to attack the target VQA models. Correspondingly, we propose a novel VQAttack model, which can iteratively generate both image and text perturbations with the designed modules: the large language model (LLM)-enhanced image attack and the cross-modal joint attack module. At each iteration, the LLM-enhanced image attack module first optimizes the latent representation-based loss to generate feature-level image perturbations. Then it incorporates an LLM to further enhance the image perturbations by optimizing the designed masked answer anti-recovery loss. The cross-modal joint attack module will be triggered at a specific iteration, which updates the image and text perturbations sequentially. Notably, the text perturbation updates are based on both the learned gradients in the word embedding space and word synonym-based substitution. Experimental results on two VQA datasets with five validated models demonstrate the effectiveness of the proposed VQAttack in the transferable attack setting, compared with state-of-the-art baselines. This work reveals a significant blind spot in the ``pre-training & fine-tuning'' paradigm on VQA tasks. Source codes will be released.

摘要: 视觉问答是计算机视觉和自然语言处理领域的一项基本任务。虽然“预训练和精调”学习范式显著提高了VQA成绩，但这种学习范式的对抗稳健性还没有被探索过。本文深入研究了一个新的问题：使用预先训练好的多模源模型来生成对抗性图文对，然后将它们转移到攻击目标的VQA模型。相应地，我们提出了一种新的VQAttack模型，该模型可以迭代地产生图像和文本扰动，并设计了两个模块：大语言模型(LLM)增强的图像攻击和跨模式联合攻击模块。在每一次迭代中，LLM增强的图像攻击模块首先优化基于潜在表示的损失，以产生特征级的图像扰动。然后，通过优化设计的抗恢复损失的蒙版答案，引入LLM来进一步增强图像扰动。跨模式联合攻击模块将在特定迭代时触发，该迭代将按顺序更新图像和文本扰动。值得注意的是，文本扰动更新基于单词嵌入空间中的学习梯度和基于单词同义词的替换。在两个VQA数据集和5个已验证模型上的实验结果表明，该算法在可转移攻击环境下具有较好的性能。这项工作揭示了VQA任务“预培训和微调”范式中的一个重大盲点。源代码将会公布。



## **19. ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages**

工具剑：分三个阶段揭开大型语言模型在工具学习中的安全问题 cs.CL

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10753v1) [paper-pdf](http://arxiv.org/pdf/2402.10753v1)

**Authors**: Junjie Ye, Sixian Li, Guanyu Li, Caishuang Huang, Songyang Gao, Yilong Wu, Qi Zhang, Tao Gui, Xuanjing Huang

**Abstract**: Tool learning is widely acknowledged as a foundational approach or deploying large language models (LLMs) in real-world scenarios. While current research primarily emphasizes leveraging tools to augment LLMs, it frequently neglects emerging safety considerations tied to their application. To fill this gap, we present $ToolSword$, a comprehensive framework dedicated to meticulously investigating safety issues linked to LLMs in tool learning. Specifically, ToolSword delineates six safety scenarios for LLMs in tool learning, encompassing $malicious$ $queries$ and $jailbreak$ $attacks$ in the input stage, $noisy$ $misdirection$ and $risky$ $cues$ in the execution stage, and $harmful$ $feedback$ and $error$ $conflicts$ in the output stage. Experiments conducted on 11 open-source and closed-source LLMs reveal enduring safety challenges in tool learning, such as handling harmful queries, employing risky tools, and delivering detrimental feedback, which even GPT-4 is susceptible to. Moreover, we conduct further studies with the aim of fostering research on tool learning safety. The data is released in https://github.com/Junjie-Ye/ToolSword.

摘要: 工具学习被广泛认为是在现实世界场景中部署大型语言模型(LLM)的基本方法。虽然目前的研究主要强调利用工具来增强LLM，但它往往忽略了与其应用相关的新出现的安全考虑。为了填补这一空白，我们提出了$ToolSword$，这是一个全面的框架，致力于在工具学习中仔细调查与LLM相关的安全问题。具体地说，Tool Sword描述了LLM在工具学习中的六个安全场景，包括输入阶段的$恶意$$查询$和$越狱$$攻击$，执行阶段的$嘈杂$$误导$和$风险$$提示$，以及输出阶段的$有害$$反馈$和$错误$$冲突$。在11个开源和封闭源代码的LLM上进行的实验表明，工具学习中存在持久的安全挑战，例如处理有害的查询、使用危险的工具以及提供有害的反馈，这些都是GPT-4容易受到的。此外，我们还进行了进一步的研究，旨在促进对工具学习安全性的研究。数据以https://github.com/Junjie-Ye/ToolSword.格式发布



## **20. Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks**

视觉-LLM可以通过自我生成的排版攻击来愚弄自己 cs.CV

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.00626v2) [paper-pdf](http://arxiv.org/pdf/2402.00626v2)

**Authors**: Maan Qraitem, Nazia Tasnim, Piotr Teterwak, Kate Saenko, Bryan A. Plummer

**Abstract**: Typographic Attacks, which involve pasting misleading text onto an image, were noted to harm the performance of Vision-Language Models like CLIP. However, the susceptibility of recent Large Vision-Language Models to these attacks remains understudied. Furthermore, prior work's Typographic attacks against CLIP randomly sample a misleading class from a predefined set of categories. However, this simple strategy misses more effective attacks that exploit LVLM(s) stronger language skills. To address these issues, we first introduce a benchmark for testing Typographic attacks against LVLM(s). Moreover, we introduce two novel and more effective \textit{Self-Generated} attacks which prompt the LVLM to generate an attack against itself: 1) Class Based Attack where the LVLM (e.g. LLaVA) is asked which deceiving class is most similar to the target class and 2) Descriptive Attacks where a more advanced LVLM (e.g. GPT4-V) is asked to recommend a Typographic attack that includes both a deceiving class and description. Using our benchmark, we uncover that Self-Generated attacks pose a significant threat, reducing LVLM(s) classification performance by up to 33\%. We also uncover that attacks generated by one model (e.g. GPT-4V or LLaVA) are effective against the model itself and other models like InstructBLIP and MiniGPT4. Code: \url{https://github.com/mqraitem/Self-Gen-Typo-Attack}

摘要: 排版攻击，包括将误导性的文本粘贴到图像上，被认为会损害像CLIP这样的视觉语言模型的性能。然而，最近的大型视觉语言模型对这些攻击的易感性仍未得到充分的研究。此外，以前的工作对CLIP的排版攻击从预定义的类别集中随机抽样一个误导性的类别。然而，这种简单的策略错过了更有效的攻击，利用了S更强的语言技能。为了解决这些问题，我们首先引入了一个基准测试，用于测试针对LVLM(S)的排版攻击。此外，我们引入了两种新的更有效的自生成攻击，它们促使LVLM生成针对自身的攻击：1)基于类的攻击，其中询问LVLM(例如LLaVA)哪个欺骗类与目标类最相似；2)描述性攻击，其中更高级的LVLM(例如GPT4-V)被要求推荐既包括欺骗性类又包括描述的排版攻击。使用我们的基准测试，我们发现自生成的攻击构成了严重的威胁，使S的LVLM分类性能降低了33%。我们还发现，一个模型(例如GPT-4V或LLaVA)生成的攻击对该模型本身以及InstructBLIP和MiniGPT4等其他模型都有效。代码：\url{https://github.com/mqraitem/Self-Gen-Typo-Attack}



## **21. Universal Vulnerabilities in Large Language Models: Backdoor Attacks for In-context Learning**

大型语言模型中的普遍漏洞：情景学习的后门攻击 cs.CL

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2401.05949v4) [paper-pdf](http://arxiv.org/pdf/2401.05949v4)

**Authors**: Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Fengjun Pan, Jinming Wen

**Abstract**: In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we design a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning demonstration prompts, which can make models behave in alignment with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 180B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models.

摘要: 情境学习是一种弥合预训练和微调之间差距的范例，在几个NLP任务中表现出很高的效率，特别是在少数情况下。尽管被广泛应用，但上下文学习容易受到恶意攻击。在这项工作中，我们提出了关于这种模式的安全问题。我们的研究表明，攻击者可以通过毒化演示上下文来操纵大型语言模型的行为，而无需对模型进行微调。具体来说，我们设计了一种新的后门攻击方法，名为ICLAttack，针对基于上下文学习的大型语言模型。我们的方法包括两种类型的攻击：中毒演示示例和中毒演示提示，这可以使模型的行为与预定义的意图保持一致。ICLAttack不需要额外的微调来植入后门，因此保留了模型的通用性。此外，中毒的例子被正确标记，增强了我们攻击方法的自然隐蔽性。在几个语言模型上的广泛实验结果，大小从1.3B到180B参数，证明了我们的攻击方法的有效性，OPT模型上三个数据集的平均攻击成功率高达95.0%。



## **22. Humans or LLMs as the Judge? A Study on Judgement Biases**

人类还是LLMS当法官？关于判断偏差的研究 cs.CL

19 pages

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.10669v2) [paper-pdf](http://arxiv.org/pdf/2402.10669v2)

**Authors**: Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, Benyou Wang

**Abstract**: Adopting human and large language models (LLM) as judges (\textit{a.k.a} human- and LLM-as-a-judge) for evaluating the performance of existing LLMs has recently gained attention. Nonetheless, this approach concurrently introduces potential biases from human and LLM judges, questioning the reliability of the evaluation results. In this paper, we propose a novel framework for investigating 5 types of biases for LLM and human judges. We curate a dataset with 142 samples referring to the revised Bloom's Taxonomy and conduct thousands of human and LLM evaluations. Results show that human and LLM judges are vulnerable to perturbations to various degrees, and that even the most cutting-edge judges possess considerable biases. We further exploit their weakness and conduct attacks on LLM judges. We hope that our work can notify the community of the vulnerability of human- and LLM-as-a-judge against perturbations, as well as the urgency of developing robust evaluation systems.

摘要: 采用人类和大语言模型(LLM)作为评判者(即人类和LLM作为评判者)来评价现有LLMS的性能最近得到了关注。尽管如此，这种方法同时引入了来自人类和LLM评委的潜在偏见，质疑评估结果的可靠性。在这篇文章中，我们提出了一个新的框架来研究5种类型的偏见，为LLM和人类法官。我们整理了一个包含142个样本的数据集，参考修订后的Bloom分类，并进行了数千次人类和LLM评估。结果表明，人类和LLM法官都不同程度地容易受到扰动的影响，即使是最尖端的法官也有相当大的偏见。我们进一步利用他们的弱点，对法律系法官进行攻击。我们希望我们的工作能够告知社会，人类和LLM作为法官面对扰动的脆弱性，以及制定强有力的评估系统的紧迫性。



## **23. Jailbreaking Proprietary Large Language Models using Word Substitution Cipher**

使用单词替换密码的越狱专有大型语言模型 cs.CL

15 pages

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10601v1) [paper-pdf](http://arxiv.org/pdf/2402.10601v1)

**Authors**: Divij Handa, Advait Chirmule, Bimal Gajera, Chitta Baral

**Abstract**: Large Language Models (LLMs) are aligned to moral and ethical guidelines but remain susceptible to creative prompts called Jailbreak that can bypass the alignment process. However, most jailbreaking prompts contain harmful questions in the natural language (mainly English), which can be detected by the LLM themselves. In this paper, we present jailbreaking prompts encoded using cryptographic techniques. We first present a pilot study on the state-of-the-art LLM, GPT-4, in decoding several safe sentences that have been encrypted using various cryptographic techniques and find that a straightforward word substitution cipher can be decoded most effectively. Motivated by this result, we use this encoding technique for writing jailbreaking prompts. We present a mapping of unsafe words with safe words and ask the unsafe question using these mapped words. Experimental results show an attack success rate (up to 59.42%) of our proposed jailbreaking approach on state-of-the-art proprietary models including ChatGPT, GPT-4, and Gemini-Pro. Additionally, we discuss the over-defensiveness of these models. We believe that our work will encourage further research in making these LLMs more robust while maintaining their decoding capabilities.

摘要: 大型语言模型(LLM)符合道德和伦理准则，但仍然容易受到称为越狱的创造性提示的影响，这些提示可以绕过匹配过程。然而，大多数越狱提示包含自然语言(主要是英语)的有害问题，LLM自己可以检测到这些问题。在本文中，我们提出了使用密码技术编码的越狱提示。我们首先介绍了目前最先进的LLM，GPT-4，在解码几个使用各种密码技术加密的安全语句时的初步研究，发现直接的单词替换密码可以被最有效地解码。受这一结果的启发，我们使用这种编码技术来编写越狱提示。我们给出了不安全词和安全词的映射，并使用这些映射的词提出了不安全的问题。实验结果表明，我们提出的越狱方法在ChatGPT、GPT-4和Gemini-Pro等最先进的专有机型上的攻击成功率高达59.42%。此外，我们还讨论了这些模型的过度防御。我们相信，我们的工作将鼓励进一步的研究，使这些LLM在保持其解码能力的同时更加健壮。



## **24. Text Embedding Inversion Security for Multilingual Language Models**

多语言模型的文本嵌入反转安全 cs.CL

18 pages, 17 Tables, 6 Figures

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2401.12192v2) [paper-pdf](http://arxiv.org/pdf/2401.12192v2)

**Authors**: Yiyi Chen, Heather Lent, Johannes Bjerva

**Abstract**: Textual data is often represented as realnumbered embeddings in NLP, particularly with the popularity of large language models (LLMs) and Embeddings as a Service (EaaS). However, storing sensitive information as embeddings can be vulnerable to security breaches, as research shows that text can be reconstructed from embeddings, even without knowledge of the underlying model. While defence mechanisms have been explored, these are exclusively focused on English, leaving other languages vulnerable to attacks. This work explores LLM security through multilingual embedding inversion. We define the problem of black-box multilingual and cross-lingual inversion attacks, and thoroughly explore their potential implications. Our findings suggest that multilingual LLMs may be more vulnerable to inversion attacks, in part because English based defences may be ineffective. To alleviate this, we propose a simple masking defense effective for both monolingual and multilingual models. This study is the first to investigate multilingual inversion attacks, shedding light on the differences in attacks and defenses across monolingual and multilingual settings.

摘要: 文本数据通常在NLP中表示为重新编号的嵌入，特别是随着大型语言模型(LLM)和嵌入即服务(EaaS)的流行。然而，将敏感信息存储为嵌入可能容易受到安全漏洞的攻击，因为研究表明，即使不知道底层模型，也可以从嵌入中重构文本。虽然已经探索了防御机制，但这些机制完全集中在英语上，使其他语言容易受到攻击。该工作通过多语言嵌入倒置来探索LLM安全性。我们定义了黑盒多语言和跨语言倒置攻击的问题，并深入探讨了它们的潜在含义。我们的发现表明，多语言LLM可能更容易受到倒置攻击，部分原因是基于英语的防御可能无效。为了缓解这一问题，我们提出了一种简单的掩蔽防御方法，既适用于单语言模型，也适用于多语言模型。这项研究是对多语言倒置攻击的第一次调查，揭示了单语和多语环境下攻击和防御的差异。



## **25. Zero-shot sampling of adversarial entities in biomedical question answering**

生物医学问答中对抗性实体的零命中抽样 cs.CL

20 pages incl. appendix, under review

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10527v1) [paper-pdf](http://arxiv.org/pdf/2402.10527v1)

**Authors**: R. Patrick Xian, Alex J. Lee, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. In high-stakes and knowledge-intensive tasks, understanding model vulnerabilities is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples in natural language processing tasks raises questions about their potential guises in other settings. Here, we propose a powerscaled distance-weighted sampling scheme in embedding space to discover diverse adversarial entities as distractors. We demonstrate its advantage over random sampling in adversarial question answering on biomedical topics. Our approach enables the exploration of different regions on the attack surface, which reveals two regimes of adversarial entities that markedly differ in their characteristics. Moreover, we show that the attacks successfully manipulate token-wise Shapley value explanations, which become deceptive in the adversarial setting. Our investigations illustrate the brittleness of domain knowledge in LLMs and reveal a shortcoming of standard evaluations for high-capacity models.

摘要: 大型语言模型(LLM)中参数领域知识的不断深入推动了它们在现实世界应用程序中的快速部署。在高风险和知识密集型任务中，了解模型漏洞对于量化模型预测的可信度和规范其使用至关重要。最近在自然语言处理任务中发现了命名实体作为对抗性例子，这引发了人们对它们在其他环境中潜在伪装的质疑。在这里，我们提出了一种嵌入空间中的加权距离加权抽样方案，以发现不同的敌意实体作为分心者。在生物医学主题的对抗性问答中，我们展示了它比随机抽样的优势。我们的方法能够探索攻击面上的不同区域，这揭示了两个在特征上明显不同的敌对实体制度。此外，我们还证明了攻击成功地操纵了令牌Shapley值解释，这在对抗性环境下变得具有欺骗性。我们的研究表明了LLMS中领域知识的脆性，并揭示了大容量模型的标准评估的缺陷。



## **26. A Trembling House of Cards? Mapping Adversarial Attacks against Language Agents**

颤抖的纸牌屋？映射针对语言代理的对抗性攻击 cs.CL

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.10196v1) [paper-pdf](http://arxiv.org/pdf/2402.10196v1)

**Authors**: Lingbo Mo, Zeyi Liao, Boyuan Zheng, Yu Su, Chaowei Xiao, Huan Sun

**Abstract**: Language agents powered by large language models (LLMs) have seen exploding development. Their capability of using language as a vehicle for thought and communication lends an incredible level of flexibility and versatility. People have quickly capitalized on this capability to connect LLMs to a wide range of external components and environments: databases, tools, the Internet, robotic embodiment, etc. Many believe an unprecedentedly powerful automation technology is emerging. However, new automation technologies come with new safety risks, especially for intricate systems like language agents. There is a surprisingly large gap between the speed and scale of their development and deployment and our understanding of their safety risks. Are we building a house of cards? In this position paper, we present the first systematic effort in mapping adversarial attacks against language agents. We first present a unified conceptual framework for agents with three major components: Perception, Brain, and Action. Under this framework, we present a comprehensive discussion and propose 12 potential attack scenarios against different components of an agent, covering different attack strategies (e.g., input manipulation, adversarial demonstrations, jailbreaking, backdoors). We also draw connections to successful attack strategies previously applied to LLMs. We emphasize the urgency to gain a thorough understanding of language agent risks before their widespread deployment.

摘要: 由大型语言模型(LLM)驱动的语言代理经历了爆炸性的发展。他们将语言作为思维和交流的媒介的能力，带来了令人难以置信的灵活性和多功能性。人们迅速利用这一能力将LLMS连接到各种外部组件和环境：数据库、工具、互联网、机器人化身等。许多人认为，一种前所未有的强大自动化技术正在出现。然而，新的自动化技术也伴随着新的安全风险，特别是对于语言代理这样复杂的系统。它们的发展和部署的速度和规模与我们对其安全风险的理解之间存在着令人惊讶的巨大差距。我们是在建造一座纸牌房子吗？在这份立场文件中，我们提出了第一次系统地绘制针对语言代理的对抗性攻击的努力。我们首先提出了一个统一的概念框架，包括三个主要组成部分：感知、大脑和行动。在这个框架下，我们对代理的不同组件进行了全面的讨论，并提出了12种潜在的攻击方案，涵盖了不同的攻击策略(例如，输入操纵、对抗性演示、越狱、后门)。我们还将其与以前应用于LLM的成功攻击策略联系起来。我们强调，在广泛部署语言诱导剂风险之前，迫切需要彻底了解这些风险。



## **27. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09132v2) [paper-pdf](http://arxiv.org/pdf/2402.09132v2)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型因其强大的语言生成能力而引起了广泛的关注，为工业和研究提供了巨大的潜力。虽然之前的研究已经深入研究了LLMS的安全和隐私问题，但这些模型在多大程度上可以表现出敌对行为，仍然很大程度上还没有被探索。针对这一差距，我们调查了常见的公开可用的LLM是否具有固有的能力来扰乱文本样本以愚弄安全措施，即所谓的对抗性示例攻击。更具体地说，我们调查LLM是否天生就能够从良性样本中制作敌意示例，以愚弄现有的安全Rail。我们的实验集中在仇恨语音检测上，实验表明，LLMS成功地发现了敌意扰动，有效地破坏了仇恨语音检测系统。我们的发现对依赖LLMS的(半)自治系统具有重大影响，突显了它们与现有系统和安全措施相互作用的潜在挑战。



## **28. Rapid Adoption, Hidden Risks: The Dual Impact of Large Language Model Customization**

快速采用，隐藏风险：大型语言模型定制的双重影响 cs.CR

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09179v2) [paper-pdf](http://arxiv.org/pdf/2402.09179v2)

**Authors**: Rui Zhang, Hongwei Li, Rui Wen, Wenbo Jiang, Yuan Zhang, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The increasing demand for customized Large Language Models (LLMs) has led to the development of solutions like GPTs. These solutions facilitate tailored LLM creation via natural language prompts without coding. However, the trustworthiness of third-party custom versions of LLMs remains an essential concern. In this paper, we propose the first instruction backdoor attacks against applications integrated with untrusted customized LLMs (e.g., GPTs). Specifically, these attacks embed the backdoor into the custom version of LLMs by designing prompts with backdoor instructions, outputting the attacker's desired result when inputs contain the pre-defined triggers. Our attack includes 3 levels of attacks: word-level, syntax-level, and semantic-level, which adopt different types of triggers with progressive stealthiness. We stress that our attacks do not require fine-tuning or any modification to the backend LLMs, adhering strictly to GPTs development guidelines. We conduct extensive experiments on 4 prominent LLMs and 5 benchmark text classification datasets. The results show that our instruction backdoor attacks achieve the desired attack performance without compromising utility. Additionally, we propose an instruction-ignoring defense mechanism and demonstrate its partial effectiveness in mitigating such attacks. Our findings highlight the vulnerability and the potential risks of LLM customization such as GPTs.

摘要: 对定制的大型语言模型(LLM)的需求日益增长，导致了GPTS等解决方案的开发。这些解决方案无需编码即可通过自然语言提示实现定制的LLM创建。然而，第三方定制版本的LLMS的可信性仍然是一个关键问题。在本文中，我们提出了针对集成了不可信任的定制LLM的应用程序(例如GPT)的第一指令后门攻击。具体地说，这些攻击通过设计带有后门指令的提示将后门嵌入到LLMS的自定义版本中，并在输入包含预定义触发器时输出攻击者所需的结果。我们的攻击包括词级、句法级和语义级三个级别的攻击，它们采用了不同类型的触发器，具有渐进的隐蔽性。我们强调，我们的攻击不需要对后端LLM进行微调或任何修改，严格遵守GPTS开发指南。我们在4个重要的LLMS和5个基准文本分类数据集上进行了大量的实验。结果表明，指令后门攻击在不影响效用的情况下达到了预期的攻击性能。此外，我们还提出了一种忽略指令的防御机制，并证明了其在缓解此类攻击方面的部分有效性。我们的发现突出了LLM定制(如GPTS)的脆弱性和潜在风险。



## **29. AbuseGPT: Abuse of Generative AI ChatBots to Create Smishing Campaigns**

AbuseGPT：滥用多产的AI聊天机器人创建Smish运动 cs.CR

6 pages, 12 figures, published in ISDFS 2024

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09728v1) [paper-pdf](http://arxiv.org/pdf/2402.09728v1)

**Authors**: Ashfak Md Shibli, Mir Mehedi A. Pritom, Maanak Gupta

**Abstract**: SMS phishing, also known as "smishing", is a growing threat that tricks users into disclosing private information or clicking into URLs with malicious content through fraudulent mobile text messages. In recent past, we have also observed a rapid advancement of conversational generative AI chatbot services (e.g., OpenAI's ChatGPT, Google's BARD), which are powered by pre-trained large language models (LLMs). These AI chatbots certainly have a lot of utilities but it is not systematically understood how they can play a role in creating threats and attacks. In this paper, we propose AbuseGPT method to show how the existing generative AI-based chatbot services can be exploited by attackers in real world to create smishing texts and eventually lead to craftier smishing campaigns. To the best of our knowledge, there is no pre-existing work that evidently shows the impacts of these generative text-based models on creating SMS phishing. Thus, we believe this study is the first of its kind to shed light on this emerging cybersecurity threat. We have found strong empirical evidences to show that attackers can exploit ethical standards in the existing generative AI-based chatbot services by crafting prompt injection attacks to create newer smishing campaigns. We also discuss some future research directions and guidelines to protect the abuse of generative AI-based services and safeguard users from smishing attacks.

摘要: 短信钓鱼是一种日益增长的威胁，它诱使用户通过欺诈性手机短信泄露私人信息或点击含有恶意内容的URL。在最近的过去，我们还观察到会话生成式AI聊天机器人服务(例如，OpenAI的ChatGPT、Google的Bard)的快速发展，这些服务由预先训练的大型语言模型(LLM)提供支持。这些人工智能聊天机器人当然有很多实用程序，但人们并不系统地了解它们如何在制造威胁和攻击方面发挥作用。在本文中，我们提出了AbuseGPT方法来展示现有的基于AI的生成性聊天机器人服务如何被现实世界中的攻击者利用来创建恶意文本，并最终导致更巧妙的恶意攻击活动。就我们所知，没有任何预先存在的工作可以明显地表明这些基于文本的生成性模型对创建短信钓鱼的影响。因此，我们认为这项研究是第一次揭示这一新出现的网络安全威胁。我们发现了强有力的经验证据表明，攻击者可以利用现有基于人工智能的生成性聊天机器人服务中的道德标准，通过手工制作快速注入攻击来创建更新的气味攻击。我们还讨论了一些未来的研究方向和指导方针，以保护基于人工智能的生成性服务的滥用，保护用户免受嗅觉攻击。



## **30. Detecting Phishing Sites Using ChatGPT**

使用ChatGPT检测钓鱼网站 cs.CR

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2306.05816v2) [paper-pdf](http://arxiv.org/pdf/2306.05816v2)

**Authors**: Takashi Koide, Naoki Fukushi, Hiroki Nakano, Daiki Chiba

**Abstract**: The emergence of Large Language Models (LLMs), including ChatGPT, is having a significant impact on a wide range of fields. While LLMs have been extensively researched for tasks such as code generation and text synthesis, their application in detecting malicious web content, particularly phishing sites, has been largely unexplored. To combat the rising tide of cyber attacks due to the misuse of LLMs, it is important to automate detection by leveraging the advanced capabilities of LLMs.   In this paper, we propose a novel system called ChatPhishDetector that utilizes LLMs to detect phishing sites. Our system involves leveraging a web crawler to gather information from websites, generating prompts for LLMs based on the crawled data, and then retrieving the detection results from the responses generated by the LLMs. The system enables us to detect multilingual phishing sites with high accuracy by identifying impersonated brands and social engineering techniques in the context of the entire website, without the need to train machine learning models. To evaluate the performance of our system, we conducted experiments on our own dataset and compared it with baseline systems and several LLMs. The experimental results using GPT-4V demonstrated outstanding performance, with a precision of 98.7% and a recall of 99.6%, outperforming the detection results of other LLMs and existing systems. These findings highlight the potential of LLMs for protecting users from online fraudulent activities and have important implications for enhancing cybersecurity measures.

摘要: 大型语言模型(LLM)的出现，包括ChatGPT，正在对广泛的领域产生重大影响。虽然LLMS已经被广泛研究用于代码生成和文本合成等任务，但它们在检测恶意网络内容，特别是钓鱼网站方面的应用在很大程度上还没有被探索过。为了应对由于滥用LLMS而不断增加的网络攻击浪潮，重要的是通过利用LLMS的高级功能来实现自动检测。在本文中，我们提出了一个新的系统，称为ChatPhishDetector，它利用LLMS来检测钓鱼网站。我们的系统利用网络爬虫从网站收集信息，基于爬行的数据生成LLMS的提示，然后从LLMS生成的响应中检索检测结果。该系统通过在整个网站的上下文中识别假冒品牌和社会工程技术，使我们能够高精度地检测多语言钓鱼网站，而不需要训练机器学习模型。为了评估我们的系统的性能，我们在自己的数据集上进行了实验，并将其与基线系统和几个LLMS进行了比较。基于GPT-4V的实验结果表明，该方法具有较好的性能，准确率为98.7%，召回率为99.6%，优于其他LLMS和现有系统的检测结果。这些发现突出了小岛屿发展中国家保护用户免遭网上欺诈活动的潜力，并对加强网络安全措施具有重要意义。



## **31. PAL: Proxy-Guided Black-Box Attack on Large Language Models**

PAL：针对大型语言模型的代理引导的黑盒攻击 cs.CL

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09674v1) [paper-pdf](http://arxiv.org/pdf/2402.09674v1)

**Authors**: Chawin Sitawarin, Norman Mu, David Wagner, Alexandre Araujo

**Abstract**: Large Language Models (LLMs) have surged in popularity in recent months, but they have demonstrated concerning capabilities to generate harmful content when manipulated. While techniques like safety fine-tuning aim to minimize harmful use, recent works have shown that LLMs remain vulnerable to attacks that elicit toxic responses. In this work, we introduce the Proxy-Guided Attack on LLMs (PAL), the first optimization-based attack on LLMs in a black-box query-only setting. In particular, it relies on a surrogate model to guide the optimization and a sophisticated loss designed for real-world LLM APIs. Our attack achieves 84% attack success rate (ASR) on GPT-3.5-Turbo and 48% on Llama-2-7B, compared to 4% for the current state of the art. We also propose GCG++, an improvement to the GCG attack that reaches 94% ASR on white-box Llama-2-7B, and the Random-Search Attack on LLMs (RAL), a strong but simple baseline for query-based attacks. We believe the techniques proposed in this work will enable more comprehensive safety testing of LLMs and, in the long term, the development of better security guardrails. The code can be found at https://github.com/chawins/pal.

摘要: 近几个月来，大型语言模型(LLM)越来越受欢迎，但它们展示了人们对操纵时生成有害内容的能力的担忧。虽然安全微调等技术旨在将有害使用降至最低，但最近的研究表明，LLM仍然容易受到引发有毒反应的攻击。在这项工作中，我们引入了代理引导的LLMS攻击(PAL)，这是在黑盒仅查询环境下第一个基于优化的LLMS攻击。特别是，它依赖于代理模型来指导优化和为真实世界的LLMAPI设计的复杂损失。我们的攻击在GPT-3.5-Turbo上实现了84%的攻击成功率(ASR)，在Llama-2-7B上实现了48%的攻击成功率，而目前的技术水平为4%。我们还提出了GCG++，它是对白盒Llama-2-7B上达到94%ASR的GCG攻击的改进，以及对LLMS的随机搜索攻击(Ral)，它是一种强大但简单的基于查询攻击的基线。我们相信，这项工作中提出的技术将使LLMS能够进行更全面的安全测试，并在长期内开发出更好的安全护栏。代码可在https://github.com/chawins/pal.上找到



## **32. How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?**

大型语言模型(LLM)在城市环境中的导航安全性如何？ cs.RO

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09546v1) [paper-pdf](http://arxiv.org/pdf/2402.09546v1)

**Authors**: Congcong Wen, Jiazhao Liang, Shuaihang Yuan, Hao Huang, Yi Fang

**Abstract**: In the field of robotics and automation, navigation systems based on Large Language Models (LLMs) have recently shown impressive performance. However, the security aspects of these systems have received relatively less attention. This paper pioneers the exploration of vulnerabilities in LLM-based navigation models in urban outdoor environments, a critical area given the technology's widespread application in autonomous driving, logistics, and emergency services. Specifically, we introduce a novel Navigational Prompt Suffix (NPS) Attack that manipulates LLM-based navigation models by appending gradient-derived suffixes to the original navigational prompt, leading to incorrect actions. We conducted comprehensive experiments on an LLMs-based navigation model that employs various LLMs for reasoning. Our results, derived from the Touchdown and Map2Seq street-view datasets under both few-shot learning and fine-tuning configurations, demonstrate notable performance declines across three metrics in the face of both white-box and black-box attacks. These results highlight the generalizability and transferability of the NPS Attack, emphasizing the need for enhanced security in LLM-based navigation systems. As an initial countermeasure, we propose the Navigational Prompt Engineering (NPE) Defense strategy, concentrating on navigation-relevant keywords to reduce the impact of adversarial suffixes. While initial findings indicate that this strategy enhances navigational safety, there remains a critical need for the wider research community to develop stronger defense methods to effectively tackle the real-world challenges faced by these systems.

摘要: 在机器人学和自动化领域，基于大语言模型(LLMS)的导航系统最近表现出令人印象深刻的性能。然而，这些系统的安全方面受到的关注相对较少。本文率先探索了城市户外环境中基于LLM的导航模型的漏洞，鉴于该技术在自动驾驶、物流和紧急服务中的广泛应用，这是一个关键领域。具体地说，我们引入了一种新的导航提示后缀(NPS)攻击，该攻击通过在原始导航提示中添加梯度派生后缀来操纵基于LLM的导航模型，从而导致不正确的操作。我们在一个基于LLMS的导航模型上进行了全面的实验，该模型使用了不同的LLMS进行推理。我们的结果来自Touchdown和Map2Seq街景数据集，在少镜头学习和微调配置下，在面对白盒和黑盒攻击时，三个指标的性能都有显著下降。这些结果突出了NPS攻击的通用性和可转移性，强调了在基于LLM的导航系统中增强安全性的必要性。作为初步对策，我们提出了导航提示工程(NPE)防御策略，将重点放在与导航相关的关键字上，以减少对抗性后缀的影响。虽然初步研究结果表明，这一战略增强了导航安全，但更广泛的研究界仍然迫切需要开发更强大的防御方法，以有效地应对这些系统所面临的现实世界挑战。



## **33. Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey**

LLM会话安全的攻击、防御与评估 cs.CL

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09283v1) [paper-pdf](http://arxiv.org/pdf/2402.09283v1)

**Authors**: Zhichen Dong, Zhanhui Zhou, Chao Yang, Jing Shao, Yu Qiao

**Abstract**: Large Language Models (LLMs) are now commonplace in conversation applications. However, their risks of misuse for generating harmful responses have raised serious societal concerns and spurred recent research on LLM conversation safety. Therefore, in this survey, we provide a comprehensive overview of recent studies, covering three critical aspects of LLM conversation safety: attacks, defenses, and evaluations. Our goal is to provide a structured summary that enhances understanding of LLM conversation safety and encourages further investigation into this important subject. For easy reference, we have categorized all the studies mentioned in this survey according to our taxonomy, available at: https://github.com/niconi19/LLM-conversation-safety.

摘要: 大型语言模型（LLM）现在在会话应用程序中很常见。然而，它们被滥用产生有害反应的风险引起了严重的社会关注，并促使最近对LLM会话安全性的研究。因此，在这次调查中，我们提供了最近研究的全面概述，涵盖了LLM会话安全的三个关键方面：攻击，防御和评估。我们的目标是提供一个结构化的摘要，增强对LLM会话安全的理解，并鼓励进一步调查这一重要主题。为了便于参考，我们根据我们的分类法对本次调查中提到的所有研究进行了分类，可在https://github.com/niconi19/LLM-conversation-safety上获得。



## **34. Leveraging the Context through Multi-Round Interactions for Jailbreaking Attacks**

通过多轮互动利用上下文进行越狱攻击 cs.LG

29 pages

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09177v1) [paper-pdf](http://arxiv.org/pdf/2402.09177v1)

**Authors**: Yixin Cheng, Markos Georgopoulos, Volkan Cevher, Grigorios G. Chrysos

**Abstract**: Large Language Models (LLMs) are susceptible to Jailbreaking attacks, which aim to extract harmful information by subtly modifying the attack query. As defense mechanisms evolve, directly obtaining harmful information becomes increasingly challenging for Jailbreaking attacks. In this work, inspired by human practices of indirect context to elicit harmful information, we focus on a new attack form called Contextual Interaction Attack. The idea relies on the autoregressive nature of the generation process in LLMs. We contend that the prior context--the information preceding the attack query--plays a pivotal role in enabling potent Jailbreaking attacks. Specifically, we propose an approach that leverages preliminary question-answer pairs to interact with the LLM. By doing so, we guide the responses of the model toward revealing the 'desired' harmful information. We conduct experiments on four different LLMs and demonstrate the efficacy of this attack, which is black-box and can also transfer across LLMs. We believe this can lead to further developments and understanding of the context vector in LLMs.

摘要: 大型语言模型(LLM)容易受到越狱攻击，其目的是通过微妙地修改攻击查询来提取有害信息。随着防御机制的发展，直接获取有害信息对越狱攻击来说变得越来越具有挑战性。在这项工作中，受人类间接上下文获取有害信息的做法的启发，我们重点研究了一种新的攻击形式，称为上下文交互攻击。这一想法依赖于LLMS中生成过程的自回归性质。我们认为，先前的上下文--攻击查询之前的信息--在实现强大的越狱攻击方面发挥着关键作用。具体地说，我们提出了一种利用初步问题-答案对与LLM交互的方法。通过这样做，我们引导该模型的反应，以揭示“想要的”有害信息。我们在四个不同的LLM上进行了实验，并展示了该攻击的有效性，该攻击是黑盒的，也可以跨LLM传输。我们相信这可以导致对LLMS中的上下文向量的进一步发展和理解。



## **35. Attacking Large Language Models with Projected Gradient Descent**

用投影梯度下降攻击大型语言模型 cs.LG

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09154v1) [paper-pdf](http://arxiv.org/pdf/2402.09154v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Johannes Gasteiger, Stephan Günnemann

**Abstract**: Current LLM alignment methods are readily broken through specifically crafted adversarial prompts. While crafting adversarial prompts using discrete optimization is highly effective, such attacks typically use more than 100,000 LLM calls. This high computational cost makes them unsuitable for, e.g., quantitative analyses and adversarial training. To remedy this, we revisit Projected Gradient Descent (PGD) on the continuously relaxed input prompt. Although previous attempts with ordinary gradient-based attacks largely failed, we show that carefully controlling the error introduced by the continuous relaxation tremendously boosts their efficacy. Our PGD for LLMs is up to one order of magnitude faster than state-of-the-art discrete optimization to achieve the same devastating attack results.

摘要: 目前的LLM对齐方法很容易突破专门制作的对抗性提示。虽然使用离散优化精心编制敌意提示是非常有效的，但此类攻击通常使用超过100,000个LLM调用。这种高的计算成本使得它们不适合例如定量分析和对抗性训练。为了纠正这个问题，我们重新审视了持续放松的输入提示上的预测梯度下降(PGD)。虽然以前使用普通的基于梯度的攻击的尝试基本上都失败了，但我们表明，仔细控制连续放松带来的错误可以极大地提高它们的效率。我们针对LLMS的PGD比最先进的离散优化快一个数量级，以实现相同的毁灭性攻击结果。



## **36. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

软提示威胁：通过嵌入空间攻击开源LLM中的安全对齐和遗忘 cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09063v1) [paper-pdf](http://arxiv.org/pdf/2402.09063v1)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.

摘要: 目前LLM对抗鲁棒性的研究主要集中在自然语言空间中的离散输入操作，这些操作可以直接转移到闭源模型。然而，这种方法忽略了开源模型的稳步发展。随着开源模型在功能上的进步，确保其安全性也变得越来越重要。然而，针对利用完整模型访问的开源LLM的攻击在很大程度上仍未被探索。我们解决了这一研究空白，并提出了嵌入空间攻击，它直接攻击输入令牌的连续嵌入表示。我们发现，嵌入空间攻击规避模型对齐和触发有害行为比离散攻击或模型微调更有效。此外，我们提出了一种新的威胁模型，在unlearning的背景下，并表明嵌入空间攻击可以从多个数据集和模型的unlearned LLM中提取被删除的信息。我们的研究结果强调了嵌入空间攻击作为开源LLM中的一个重要威胁模型。触发警告：附录包含LLM生成的暴力和骚扰文本。



## **37. Prompted Contextual Vectors for Spear-Phishing Detection**

鱼叉式网络钓鱼检测的提示上下文向量 cs.LG

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.08309v2) [paper-pdf](http://arxiv.org/pdf/2402.08309v2)

**Authors**: Daniel Nahmias, Gal Engelberg, Dan Klein, Asaf Shabtai

**Abstract**: Spear-phishing attacks present a significant security challenge, with large language models (LLMs) escalating the threat by generating convincing emails and facilitating target reconnaissance. To address this, we propose a detection approach based on a novel document vectorization method that utilizes an ensemble of LLMs to create representation vectors. By prompting LLMs to reason and respond to human-crafted questions, we quantify the presence of common persuasion principles in the email's content, producing prompted contextual document vectors for a downstream supervised machine learning model. We evaluate our method using a unique dataset generated by a proprietary system that automates target reconnaissance and spear-phishing email creation. Our method achieves a 91% F1 score in identifying LLM-generated spear-phishing emails, with the training set comprising only traditional phishing and benign emails. Key contributions include an innovative document vectorization method utilizing LLM reasoning, a publicly available dataset of high-quality spear-phishing emails, and the demonstrated effectiveness of our method in detecting such emails. This methodology can be utilized for various document classification tasks, particularly in adversarial problem domains.

摘要: 鱼叉式网络钓鱼攻击是一个重大的安全挑战，大型语言模型（LLM）通过生成令人信服的电子邮件和促进目标侦察来升级威胁。为了解决这个问题，我们提出了一种检测方法的基础上，一种新的文档矢量化方法，利用一个合奏的LLM创建表示向量。通过促使LLM推理并响应人工问题，我们量化了电子邮件内容中常见说服原则的存在，为下游监督机器学习模型生成提示的上下文文档向量。我们使用专有系统生成的独特数据集来评估我们的方法，该系统可自动进行目标侦察和鱼叉式网络钓鱼电子邮件创建。我们的方法在识别LLM生成的鱼叉式网络钓鱼电子邮件方面达到了91%的F1分数，训练集仅包括传统的网络钓鱼和良性电子邮件。主要贡献包括利用LLM推理的创新文档矢量化方法，高质量鱼叉式网络钓鱼电子邮件的公开数据集，以及我们的方法在检测此类电子邮件方面的有效性。该方法可用于各种文档分类任务，特别是在对抗性问题领域。



## **38. SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding**

安全解码：通过安全感知解码防御越狱攻击 cs.CR

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.08983v1) [paper-pdf](http://arxiv.org/pdf/2402.08983v1)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, Radha Poovendran

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their token probabilities, while simultaneously attenuating the probabilities of token sequences that are aligned with the objectives of jailbreak attacks. We perform extensive experiments on five LLMs using six state-of-the-art jailbreak attacks and four benchmark datasets. Our results show that SafeDecoding significantly reduces the attack success rate and harmfulness of jailbreak attacks without compromising the helpfulness of responses to benign user queries. SafeDecoding outperforms six defense methods.

摘要: 随着大型语言模型(LLM)越来越多地集成到真实世界的应用中，如代码生成和聊天机器人辅助，人们已经做出了广泛的努力来使LLM的行为与包括安全在内的人类价值观保持一致。越狱攻击旨在挑起LLM的意外和不安全行为，仍然是LLM的重大/主要安全威胁。在本文中，我们的目标是通过引入SafeDecoding来防御LLMS的越狱攻击，SafeDecoding是一种安全感知的解码策略，用于LLMS对用户查询生成有用和无害的响应。我们开发SafeDecoding的洞察力基于这样的观察：即使代表有害内容的令牌的概率大于代表无害响应的令牌的概率，但在按概率降序对令牌进行排序后，安全免责声明仍会出现在排名最靠前的令牌中。这使我们能够通过识别安全免责声明并放大其令牌概率来缓解越狱攻击，同时降低与越狱攻击目标一致的令牌序列的概率。我们使用六个最先进的越狱攻击和四个基准数据集在五个LLM上进行了广泛的实验。我们的结果表明，SafeDecoding在不影响对良性用户查询的响应的帮助的情况下，显著降低了越狱攻击的攻击成功率和危害性。安全解码的性能超过了六种防御方法。



## **39. COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability**

冷攻击：具有隐蔽性和可控性的越狱LLMS cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08679v1) [paper-pdf](http://arxiv.org/pdf/2402.08679v1)

**Authors**: Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, Bin Hu

**Abstract**: Jailbreaks on Large language models (LLMs) have recently received increasing attention. For a comprehensive assessment of LLM safety, it is essential to consider jailbreaks with diverse attributes, such as contextual coherence and sentiment/stylistic variations, and hence it is beneficial to study controllable jailbreaking, i.e. how to enforce control on LLM attacks. In this paper, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the Energy-based Constrained Decoding with Langevin Dynamics (COLD), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios which not only cover the standard setting of generating fluent suffix attacks, but also allow us to address new controllable attack settings such as revising a user query adversarially with minimal paraphrasing, and inserting stealthy attacks in context with left-right-coherence. Our extensive experiments on various LLMs (Llama-2, Mistral, Vicuna, Guanaco, GPT-3.5) show COLD-Attack's broad applicability, strong controllability, high success rate, and attack transferability. Our code is available at https://github.com/Yu-Fangxu/COLD-Attack.

摘要: 大型语言模型(LLM)的越狱最近受到越来越多的关注。为了全面评估LLM的安全性，必须考虑具有不同属性的越狱，例如上下文连贯性和情绪/风格变化，因此研究可控越狱是有益的，即如何加强对LLM攻击的控制。在本文中，我们形式化地描述了可控攻击生成问题，并将该问题与自然语言处理的一个热门话题--可控文本生成建立了一种新的联系。基于此，我们采用了基于能量的朗之万动力学约束解码算法(COLD)，这是一种最新的、高效的可控文本生成算法，并引入了冷攻击框架，该框架可以在流畅性、隐蔽性、情感和左右一致性等各种控制要求下统一和自动化搜索敌意LLM攻击。冷攻击带来的可控性导致了不同的新越狱场景，这些场景不仅覆盖了生成流畅后缀攻击的标准设置，而且允许我们解决新的可控攻击设置，例如以最小的转述以相反的方式修改用户查询，以及以左右一致的方式在上下文中插入隐蔽攻击。我们在不同的LLMS(骆驼-2、米斯特拉尔、维库纳、瓜纳科、GPT-3.5)上的广泛实验表明，冷攻击具有广泛的适用性、较强的可控性、高成功率和攻击可转移性。我们的代码可以在https://github.com/Yu-Fangxu/COLD-Attack.上找到



## **40. Test-Time Backdoor Attacks on Multimodal Large Language Models**

多模态大型语言模型的测试时后门攻击 cs.CL

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08577v1) [paper-pdf](http://arxiv.org/pdf/2402.08577v1)

**Authors**: Dong Lu, Tianyu Pang, Chao Du, Qian Liu, Xianjun Yang, Min Lin

**Abstract**: Backdoor attacks are commonly executed by contaminating training data, such that a trigger can activate predetermined harmful effects during the test phase. In this work, we present AnyDoor, a test-time backdoor attack against multimodal large language models (MLLMs), which involves injecting the backdoor into the textual modality using adversarial test images (sharing the same universal perturbation), without requiring access to or modification of the training data. AnyDoor employs similar techniques used in universal adversarial attacks, but distinguishes itself by its ability to decouple the timing of setup and activation of harmful effects. In our experiments, we validate the effectiveness of AnyDoor against popular MLLMs such as LLaVA-1.5, MiniGPT-4, InstructBLIP, and BLIP-2, as well as provide comprehensive ablation studies. Notably, because the backdoor is injected by a universal perturbation, AnyDoor can dynamically change its backdoor trigger prompts/harmful effects, exposing a new challenge for defending against backdoor attacks. Our project page is available at https://sail-sg.github.io/AnyDoor/.

摘要: 后门攻击通常通过污染训练数据来执行，使得触发器可以在测试阶段激活预定的有害影响。在这项工作中，我们提出了AnyDoor，一种针对多模态大型语言模型（MLLM）的测试时后门攻击，它涉及使用对抗性测试图像（共享相同的通用扰动）将后门注入文本模态，而无需访问或修改训练数据。AnyDoor采用了通用对抗性攻击中使用的类似技术，但其独特之处在于能够将设置时间和有害影响的激活解耦。在我们的实验中，我们验证了AnyDoor对流行MLLM（如LLaVA-1.5、MiniGPT-4、InstructBLIP和BLIP-2）的有效性，并提供了全面的消融研究。值得注意的是，由于后门是通过通用扰动注入的，AnyDoor可以动态改变其后门触发提示/有害影响，为防御后门攻击带来了新的挑战。我们的项目页面可以在https://sail-sg.github.io/AnyDoor/上找到。



## **41. Pandora: Jailbreak GPTs by Retrieval Augmented Generation Poisoning**

Pandora：通过检索增强生成中毒越狱GPT cs.CR

6 pages

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08416v1) [paper-pdf](http://arxiv.org/pdf/2402.08416v1)

**Authors**: Gelei Deng, Yi Liu, Kailong Wang, Yuekang Li, Tianwei Zhang, Yang Liu

**Abstract**: Large Language Models~(LLMs) have gained immense popularity and are being increasingly applied in various domains. Consequently, ensuring the security of these models is of paramount importance. Jailbreak attacks, which manipulate LLMs to generate malicious content, are recognized as a significant vulnerability. While existing research has predominantly focused on direct jailbreak attacks on LLMs, there has been limited exploration of indirect methods. The integration of various plugins into LLMs, notably Retrieval Augmented Generation~(RAG), which enables LLMs to incorporate external knowledge bases into their response generation such as GPTs, introduces new avenues for indirect jailbreak attacks.   To fill this gap, we investigate indirect jailbreak attacks on LLMs, particularly GPTs, introducing a novel attack vector named Retrieval Augmented Generation Poisoning. This method, Pandora, exploits the synergy between LLMs and RAG through prompt manipulation to generate unexpected responses. Pandora uses maliciously crafted content to influence the RAG process, effectively initiating jailbreak attacks. Our preliminary tests show that Pandora successfully conducts jailbreak attacks in four different scenarios, achieving higher success rates than direct attacks, with 64.3\% for GPT-3.5 and 34.8\% for GPT-4.

摘要: 大语言模型(LLMS)已经得到了广泛的应用，并在各个领域得到了越来越多的应用。因此，确保这些模型的安全至关重要。越狱攻击操纵LLM生成恶意内容，被认为是一个严重的漏洞。虽然现有的研究主要集中在对LLM的直接越狱攻击上，但对间接方法的探索有限。将各种插件集成到LLMS中，特别是检索增强一代~(RAG)，使LLMS能够将外部知识库整合到其响应生成中，如GPT，为间接越狱攻击引入了新的途径。为了填补这一空白，我们研究了针对LLM的间接越狱攻击，特别是GPT，引入了一种新的攻击载体-检索增强生成毒化。这种名为Pandora的方法通过即时操作来利用LLMS和RAG之间的协同作用来产生意想不到的反应。Pandora使用恶意创建的内容来影响RAG过程，从而有效地发起越狱攻击。我们的初步测试表明，Pandora在四种不同的场景下成功地进行了越狱攻击，取得了比直接攻击更高的成功率，GPT-3.5和GPT-4的成功率分别为64.3%和34.8%。



## **42. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

攻坚战：如何评估越狱攻击在大型语言模型上的有效性 cs.CL

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2401.09002v2) [paper-pdf](http://arxiv.org/pdf/2401.09002v2)

**Authors**: Dong shu, Mingyu Jin, Suiyuan Zhu, Beichen Wang, Zihao Zhou, Chong Zhang, Yongfeng Zhang

**Abstract**: In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation aligns with the baseline's trend while offering a more profound and detailed assessment. We believe that by accurately evaluating the effectiveness of attack prompts in the Jailbreak task, our work lays a solid foundation for assessing a wider array of similar or even more complex tasks in the realm of prompt injection, potentially revolutionizing this field.

摘要: 在我们的研究中，我们开创了一种新的方法来评估越狱攻击对大型语言模型(如GPT-4和LLaMa2)的有效性，不同于传统的专注于健壮性的二进制评估。我们的研究引入了两个不同的评估框架：粗粒度评估和细粒度评估。每个框架使用从0到1的评分范围，提供了一个独特的视角，能够对攻击效果进行更全面和细微的评估，并使攻击者能够更好地了解他们的攻击提示。此外，我们还开发了专门为越狱任务量身定做的全面地面事实数据集。这一数据集不仅是我们当前研究的重要基准，而且还为未来的研究奠定了基础资源，使这一不断发展的领域能够进行一致和比较的分析。通过与传统评估方法的细致比较，我们发现我们的评估符合基线的趋势，同时提供了更深入和详细的评估。我们相信，通过准确评估越狱任务中攻击提示的有效性，我们的工作为评估快速注射领域中更广泛的类似甚至更复杂的任务奠定了坚实的基础，这可能会给这一领域带来革命性的变化。



## **43. PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining**

PANORAMIA：无需再培训的机器学习模型隐私审计 cs.CR

19 pages

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.09477v1) [paper-pdf](http://arxiv.org/pdf/2402.09477v1)

**Authors**: Mishaal Kazmi, Hadrien Lautraite, Alireza Akbari, Mauricio Soroco, Qiaoyue Tang, Tao Wang, Sébastien Gambs, Mathias Lécuyer

**Abstract**: We introduce a privacy auditing scheme for ML models that relies on membership inference attacks using generated data as "non-members". This scheme, which we call PANORAMIA, quantifies the privacy leakage for large-scale ML models without control of the training process or model re-training and only requires access to a subset of the training data. To demonstrate its applicability, we evaluate our auditing scheme across multiple ML domains, ranging from image and tabular data classification to large-scale language models.

摘要: 我们引入了一个隐私审计方案的ML模型，依赖于成员推理攻击使用生成的数据作为“非成员”。我们称之为PANORAMIA的这个方案量化了大规模ML模型的隐私泄露，而无需控制训练过程或模型重新训练，只需要访问训练数据的一个子集。为了证明其适用性，我们评估了我们在多个ML领域的审计方案，从图像和表格数据分类到大规模语言模型。



## **44. Certifying LLM Safety against Adversarial Prompting**

认证LLM安全对抗性认证 cs.CL

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2309.02705v3) [paper-pdf](http://arxiv.org/pdf/2309.02705v3)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击会向输入提示添加恶意令牌，以绕过LLM的安全护栏，导致其产生有害内容。在这项工作中，我们引入了Erase-and-Check，这是第一个用于防御具有可证明安全保证的对抗性提示的框架。在给定提示的情况下，我们的过程将逐个擦除令牌，并使用安全过滤器检查结果子序列。我们的安全证书保证有害提示不会因为达到一定大小的敌意攻击而被错误地标记为安全。我们用Llama 2和DistilBERT两种方法实现了安全过滤器，并比较了两种情况下的擦除和检查性能。我们防御三种攻击模式：i)对抗性后缀，其中对抗性序列被附加在有害提示的末尾；ii)对抗性插入，其中对抗性序列被插入在提示中间的任何位置；以及iii)对抗性注入，其中对抗性标记被插入在提示中的任意位置，而不一定作为连续的块。实验结果表明，该方法在对安全提示保持较好经验性能的同时，对有害提示具有较强的认证安全保障。此外，我们还提出了三种有效的经验防御：i)RandEC，一种随机化的擦除和检查版本；ii)GreedyEC，它贪婪地擦除使有害类别的Softmax得分最大化的标记；以及iii)Gradec，它使用梯度信息来优化要擦除的标记。我们证明了它们对贪婪坐标梯度(GCG)攻击算法生成的敌意提示的有效性。我们实验的代码可以在https://github.com/aounon/certified-llm-safety.上找到



## **45. PoisonedRAG: Knowledge Poisoning Attacks to Retrieval-Augmented Generation of Large Language Models**

PoisonedRAG：对大型语言模型检索增强生成的知识毒化攻击 cs.CR

Code is available at https://github.com/sleeepeer/PoisonedRAG

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07867v1) [paper-pdf](http://arxiv.org/pdf/2402.07867v1)

**Authors**: Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia

**Abstract**: Large language models (LLMs) have achieved remarkable success due to their exceptional generative capabilities. Despite their success, they also have inherent limitations such as a lack of up-to-date knowledge and hallucination. Retrieval-Augmented Generation (RAG) is a state-of-the-art technique to mitigate those limitations. In particular, given a question, RAG retrieves relevant knowledge from a knowledge database to augment the input of the LLM. For instance, the retrieved knowledge could be a set of top-k texts that are most semantically similar to the given question when the knowledge database contains millions of texts collected from Wikipedia. As a result, the LLM could utilize the retrieved knowledge as the context to generate an answer for the given question. Existing studies mainly focus on improving the accuracy or efficiency of RAG, leaving its security largely unexplored. We aim to bridge the gap in this work. Particularly, we propose PoisonedRAG , a set of knowledge poisoning attacks to RAG, where an attacker could inject a few poisoned texts into the knowledge database such that the LLM generates an attacker-chosen target answer for an attacker-chosen target question. We formulate knowledge poisoning attacks as an optimization problem, whose solution is a set of poisoned texts. Depending on the background knowledge (e.g., black-box and white-box settings) of an attacker on the RAG, we propose two solutions to solve the optimization problem, respectively. Our results on multiple benchmark datasets and LLMs show our attacks could achieve 90% attack success rates when injecting 5 poisoned texts for each target question into a database with millions of texts. We also evaluate recent defenses and our results show they are insufficient to defend against our attacks, highlighting the need for new defenses.

摘要: 大型语言模型(LLM)由于其非凡的生成能力而取得了显著的成功。尽管他们取得了成功，但他们也有内在的局限性，比如缺乏最新的知识和幻觉。检索-增强生成(RAG)是一种最先进的技术，可以缓解这些限制。特别是，给定一个问题，RAG从知识数据库中检索相关知识，以增加LLM的输入。例如，当知识数据库包含从维基百科收集的数百万文本时，检索到的知识可以是在语义上与给定问题最相似的一组top-k文本。结果，LLM可以利用检索到的知识作为上下文来生成给定问题的答案。现有的研究主要集中于提高RAG的准确性或效率，而其安全性在很大程度上还没有被探索。我们的目标是弥合这项工作中的差距。具体地说，我们提出了PoisonedRAG，这是一组针对RAG的知识毒化攻击，攻击者可以将几个有毒文本注入到知识库中，以便LLM为攻击者选择的目标问题生成攻击者选择的目标答案。我们将知识中毒攻击描述为一个优化问题，其解是一组有毒文本。根据攻击者在RAG上的背景知识(例如，黑盒和白盒设置)，我们分别提出了两种解决优化问题的方案。我们在多个基准数据集和LLMS上的实验结果表明，当每个目标问题注入5个有毒文本到数百万个文本的数据库中时，我们的攻击可以达到90%的攻击成功率。我们还评估了最近的防御，我们的结果表明，它们不足以防御我们的攻击，这突显了需要新的防御。



## **46. Do Membership Inference Attacks Work on Large Language Models?**

成员资格推理攻击在大型语言模型上有效吗？ cs.CL

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07841v1) [paper-pdf](http://arxiv.org/pdf/2402.07841v1)

**Authors**: Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi

**Abstract**: Membership inference attacks (MIAs) attempt to predict whether a particular datapoint is a member of a target model's training data. Despite extensive research on traditional machine learning models, there has been limited work studying MIA on the pre-training data of large language models (LLMs). We perform a large-scale evaluation of MIAs over a suite of language models (LMs) trained on the Pile, ranging from 160M to 12B parameters. We find that MIAs barely outperform random guessing for most settings across varying LLM sizes and domains. Our further analyses reveal that this poor performance can be attributed to (1) the combination of a large dataset and few training iterations, and (2) an inherently fuzzy boundary between members and non-members. We identify specific settings where LLMs have been shown to be vulnerable to membership inference and show that the apparent success in such settings can be attributed to a distribution shift, such as when members and non-members are drawn from the seemingly identical domain but with different temporal ranges. We release our code and data as a unified benchmark package that includes all existing MIAs, supporting future work.

摘要: 成员关系推理攻击(MIA)试图预测特定数据点是否为目标模型训练数据的成员。尽管对传统的机器学习模型进行了广泛的研究，但在大型语言模型(LLMS)的训练前数据上研究MIA的工作有限。我们在堆上训练的一套语言模型(LMS)上对MIA进行了大规模评估，参数范围从160M到12B。我们发现，对于不同的LLM大小和域，对于大多数设置，MIA的性能仅略高于随机猜测。我们的进一步分析表明，这种糟糕的性能可以归因于(1)庞大的数据集和很少的训练迭代的组合，以及(2)成员和非成员之间固有的模糊边界。我们确定了LLM被证明易受成员关系推断影响的特定设置，并表明此类设置的明显成功可以归因于分布变化，例如当成员和非成员来自看似相同的域但具有不同的时间范围时。我们将我们的代码和数据作为一个统一的基准程序包发布，其中包括所有现有的MIA，以支持未来的工作。



## **47. Universal Jailbreak Backdoors from Poisoned Human Feedback**

来自有毒人类反馈的通用越狱后门 cs.AI

Accepted as conference paper in ICLR 2024

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2311.14455v3) [paper-pdf](http://arxiv.org/pdf/2311.14455v3)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.

摘要: 来自人类反馈的强化学习(RLHF)被用来对齐大型语言模型以产生有益和无害的响应。然而，先前的工作表明，这些模型可以通过找到敌意提示来越狱，这些提示可以将模型恢复到其不一致的行为。在本文中，我们考虑了一种新的威胁，即攻击者在RLHF训练数据中下毒，以便在模型中嵌入“越狱后门”。后门将触发词嵌入到模型中，其作用类似于通用的“sudo命令”：将触发词添加到任何提示中都可以实现有害的响应，而无需搜索敌意提示。通用越狱后门比之前在语言模型上研究的后门要强大得多，我们发现使用常见的后门攻击技术来植入它们要困难得多。我们调查了RLHF中有助于其所谓的健壮性的设计决策，并发布了一个有毒模型的基准，以刺激未来对通用越狱后门的研究。



## **48. Large Language Models are Few-shot Generators: Proposing Hybrid Prompt Algorithm To Generate Webshell Escape Samples**

大型语言模型是少有的生成器：提出混合提示算法来生成WebShell逃逸示例 cs.CR

13 pages, 16 figures

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07408v1) [paper-pdf](http://arxiv.org/pdf/2402.07408v1)

**Authors**: Mingrui Ma, Lansheng Han, Chunjie Zhou

**Abstract**: The frequent occurrence of cyber-attacks has made webshell attacks and defense gradually become a research hotspot in the field of network security. However, the lack of publicly available benchmark datasets and the over-reliance on manually defined rules for webshell escape sample generation have slowed down the progress of research related to webshell escape sample generation strategies and artificial intelligence-based webshell detection algorithms. To address the drawbacks of weak webshell sample escape capabilities, the lack of webshell datasets with complex malicious features, and to promote the development of webshell detection technology, we propose the Hybrid Prompt algorithm for webshell escape sample generation with the help of large language models. As a prompt algorithm specifically developed for webshell sample generation, the Hybrid Prompt algorithm not only combines various prompt ideas including Chain of Thought, Tree of Thought, but also incorporates various components such as webshell hierarchical module and few-shot example to facilitate the LLM in learning and reasoning webshell escape strategies. Experimental results show that the Hybrid Prompt algorithm can work with multiple LLMs with excellent code reasoning ability to generate high-quality webshell samples with high Escape Rate (88.61% with GPT-4 model on VIRUSTOTAL detection engine) and Survival Rate (54.98% with GPT-4 model).

摘要: 网络攻击的频繁发生使网络外壳攻击与防御逐渐成为网络安全领域的研究热点。然而，缺乏公开可用的基准数据集，以及过度依赖人工定义的网页外壳逃逸样本生成规则，减缓了与网页外壳逃逸样本生成策略和基于人工智能的网页外壳检测算法相关的研究进展。针对网页外壳样本逃逸能力弱、缺乏具有复杂恶意特征的网页外壳数据集的不足，为推动网页外壳检测技术的发展，本文提出了基于大型语言模型的网页外壳逃逸样本混合提示生成算法。混合提示算法是专门为Web外壳样本生成而开发的一种提示算法，它不仅结合了链式、树型等多种提示思想，还加入了Web外壳分层模块、少镜头实例等多种组件，方便了LLM在学习和推理Web外壳逃逸策略方面的应用。实验结果表明，混合提示算法能够与具有良好代码推理能力的多个LLMS协同工作，生成高逃逸率(在VirusTotal检测引擎上的GPT-4模型为88.61%)和存活率(GPT-4模型为54.98%)的高质量Web外壳样本。



## **49. All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks**

万事俱备：越狱攻击的简单黑匣子方法 cs.CL

12 pages, 4 figures, 3 tables

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2401.09798v3) [paper-pdf](http://arxiv.org/pdf/2401.09798v3)

**Authors**: Kazuhiro Takemoto

**Abstract**: Large Language Models (LLMs), such as ChatGPT, encounter `jailbreak' challenges, wherein safeguards are circumvented to generate ethically harmful prompts. This study introduces a straightforward black-box method for efficiently crafting jailbreak prompts, addressing the significant complexity and computational costs associated with conventional methods. Our technique iteratively transforms harmful prompts into benign expressions directly utilizing the target LLM, predicated on the hypothesis that LLMs can autonomously generate expressions that evade safeguards. Through experiments conducted with ChatGPT (GPT-3.5 and GPT-4) and Gemini-Pro, our method consistently achieved an attack success rate exceeding 80% within an average of five iterations for forbidden questions and proved robust against model updates. The jailbreak prompts generated were not only naturally-worded and succinct but also challenging to defend against. These findings suggest that the creation of effective jailbreak prompts is less complex than previously believed, underscoring the heightened risk posed by black-box jailbreak attacks.

摘要: 大型语言模型(LLM)，如ChatGPT，会遇到“越狱”挑战，在这种情况下，安全措施会被绕过，以生成道德上有害的提示。这项研究介绍了一种简单的黑盒方法，用于高效地制作越狱提示，解决了与传统方法相关的显著复杂性和计算成本。我们的技术直接利用目标LLM迭代地将有害提示转换为良性表情，其基础是假设LLM可以自主生成逃避安全保护的表情。通过对ChatGPT(GPT-3.5和GPT-4)和Gemini-Pro的实验，我们的方法对于禁止问题在平均5次迭代内一致地获得了超过80%的攻击成功率，并被证明对模型更新具有健壮性。产生的越狱提示不仅措辞自然、简洁，而且对防御也具有挑战性。这些发现表明，创建有效的越狱提示并不像之前认为的那样复杂，这突显了黑匣子越狱攻击带来的更高风险。



## **50. Whispers in the Machine: Confidentiality in LLM-integrated Systems**

机器中的窃窃私语：LLM集成系统的机密性 cs.CR

**SubmitDate**: 2024-02-10    [abs](http://arxiv.org/abs/2402.06922v1) [paper-pdf](http://arxiv.org/pdf/2402.06922v1)

**Authors**: Jonathan Evertz, Merlin Chlosta, Lea Schönherr, Thorsten Eisenhofer

**Abstract**: Large Language Models (LLMs) are increasingly integrated with external tools. While these integrations can significantly improve the functionality of LLMs, they also create a new attack surface where confidential data may be disclosed between different components. Specifically, malicious tools can exploit vulnerabilities in the LLM itself to manipulate the model and compromise the data of other services, raising the question of how private data can be protected in the context of LLM integrations.   In this work, we provide a systematic way of evaluating confidentiality in LLM-integrated systems. For this, we formalize a "secret key" game that can capture the ability of a model to conceal private information. This enables us to compare the vulnerability of a model against confidentiality attacks and also the effectiveness of different defense strategies. In this framework, we evaluate eight previously published attacks and four defenses. We find that current defenses lack generalization across attack strategies. Building on this analysis, we propose a method for robustness fine-tuning, inspired by adversarial training. This approach is effective in lowering the success rate of attackers and in improving the system's resilience against unknown attacks.

摘要: 大型语言模型(LLM)越来越多地与外部工具集成。虽然这些集成可以显著改进LLMS的功能，但它们也创建了一个新的攻击面，其中机密数据可能会在不同的组件之间泄露。具体地说，恶意工具可以利用LLM本身的漏洞来操纵模型并危害其他服务的数据，这引发了如何在LLM集成的上下文中保护私有数据的问题。在这项工作中，我们提供了一种系统的方法来评估LLM集成系统的机密性。为此，我们形式化了一个“密钥”游戏，它可以捕获模型隐藏私人信息的能力。这使我们能够比较模型对机密性攻击的脆弱性以及不同防御策略的有效性。在这个框架中，我们评估了之前发布的八种攻击和四种防御措施。我们发现，目前的防御缺乏对攻击策略的概括性。在此分析的基础上，我们提出了一种受对手训练启发的健壮性微调方法。这种方法在降低攻击者的成功率和提高系统对未知攻击的弹性方面是有效的。



