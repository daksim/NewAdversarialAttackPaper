# Latest Large Language Model Attack Papers
**update at 2024-03-25 09:33:03**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. LogPrécis: Unleashing Language Models for Automated Malicious Log Analysis**

cs.CR

18 pages, Computer&Security  (https://www.sciencedirect.com/science/article/pii/S0167404824001068), code  available at https://github.com/SmartData-Polito/logprecis, models available  at https://huggingface.co/SmartDataPolito

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2307.08309v3) [paper-pdf](http://arxiv.org/pdf/2307.08309v3)

**Authors**: Matteo Boffa, Rodolfo Vieira Valentim, Luca Vassio, Danilo Giordano, Idilio Drago, Marco Mellia, Zied Ben Houidi

**Abstract**: The collection of security-related logs holds the key to understanding attack behaviors and diagnosing vulnerabilities. Still, their analysis remains a daunting challenge. Recently, Language Models (LMs) have demonstrated unmatched potential in understanding natural and programming languages. The question arises whether and how LMs could be also useful for security experts since their logs contain intrinsically confused and obfuscated information. In this paper, we systematically study how to benefit from the state-of-the-art in LM to automatically analyze text-like Unix shell attack logs. We present a thorough design methodology that leads to LogPr\'ecis. It receives as input raw shell sessions and automatically identifies and assigns the attacker tactic to each portion of the session, i.e., unveiling the sequence of the attacker's goals. We demonstrate LogPr\'ecis capability to support the analysis of two large datasets containing about 400,000 unique Unix shell attacks. LogPr\'ecis reduces them into about 3,000 fingerprints, each grouping sessions with the same sequence of tactics. The abstraction it provides lets the analyst better understand attacks, identify fingerprints, detect novelty, link similar attacks, and track families and mutations. Overall, LogPr\'ecis, released as open source, paves the way for better and more responsive defense against cyberattacks.



## **2. Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images**

cs.CV

Accepted by ICLR 2024

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2401.11170v2) [paper-pdf](http://arxiv.org/pdf/2401.11170v2)

**Authors**: Kuofeng Gao, Yang Bai, Jindong Gu, Shu-Tao Xia, Philip Torr, Zhifeng Li, Wei Liu

**Abstract**: Large vision-language models (VLMs) such as GPT-4 have achieved exceptional performance across various multi-modal tasks. However, the deployment of VLMs necessitates substantial energy consumption and computational resources. Once attackers maliciously induce high energy consumption and latency time (energy-latency cost) during inference of VLMs, it will exhaust computational resources. In this paper, we explore this attack surface about availability of VLMs and aim to induce high energy-latency cost during inference of VLMs. We find that high energy-latency cost during inference of VLMs can be manipulated by maximizing the length of generated sequences. To this end, we propose verbose images, with the goal of crafting an imperceptible perturbation to induce VLMs to generate long sentences during inference. Concretely, we design three loss objectives. First, a loss is proposed to delay the occurrence of end-of-sequence (EOS) token, where EOS token is a signal for VLMs to stop generating further tokens. Moreover, an uncertainty loss and a token diversity loss are proposed to increase the uncertainty over each generated token and the diversity among all tokens of the whole generated sequence, respectively, which can break output dependency at token-level and sequence-level. Furthermore, a temporal weight adjustment algorithm is proposed, which can effectively balance these losses. Extensive experiments demonstrate that our verbose images can increase the length of generated sequences by 7.87 times and 8.56 times compared to original images on MS-COCO and ImageNet datasets, which presents potential challenges for various applications. Our code is available at https://github.com/KuofengGao/Verbose_Images.



## **3. Self-Guard: Empower the LLM to Safeguard Itself**

cs.CL

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2310.15851v2) [paper-pdf](http://arxiv.org/pdf/2310.15851v2)

**Authors**: Zezhong Wang, Fangkai Yang, Lu Wang, Pu Zhao, Hongru Wang, Liang Chen, Qingwei Lin, Kam-Fai Wong

**Abstract**: The jailbreak attack can bypass the safety measures of a Large Language Model (LLM), generating harmful content. This misuse of LLM has led to negative societal consequences. Currently, there are two main approaches to address jailbreak attacks: safety training and safeguards. Safety training focuses on further training LLM to enhance its safety. On the other hand, safeguards involve implementing external models or filters to prevent harmful outputs. However, safety training has constraints in its ability to adapt to new attack types and often leads to a drop in model performance. Safeguards have proven to be of limited help. To tackle these issues, we propose a novel approach called Self-Guard, which combines the strengths of both safety methods. Self-Guard includes two stages. In the first stage, we enhance the model's ability to assess harmful content, and in the second stage, we instruct the model to consistently perform harmful content detection on its own responses. The experiment has demonstrated that Self-Guard is robust against jailbreak attacks. In the bad case analysis, we find that LLM occasionally provides harmless responses to harmful queries. Additionally, we evaluated the general capabilities of the LLM before and after safety training, providing evidence that Self-Guard does not result in the LLM's performance degradation. In sensitivity tests, Self-Guard not only avoids inducing over-sensitivity in LLM but also can even mitigate this issue.



## **4. Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation**

cs.CV

Project Page: https://gyhdog99.github.io/projects/ecso/

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.09572v2) [paper-pdf](http://arxiv.org/pdf/2403.09572v2)

**Authors**: Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang

**Abstract**: Multimodal large language models (MLLMs) have shown impressive reasoning abilities, which, however, are also more vulnerable to jailbreak attacks than their LLM predecessors. Although still capable of detecting unsafe responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs can be easily bypassed due to the introduction of image features. To construct robust MLLMs, we propose ECSO(Eyes Closed, Safety On), a novel training-free protecting approach that exploits the inherent safety awareness of MLLMs, and generates safer responses via adaptively transforming unsafe images into texts to activate intrinsic safety mechanism of pre-aligned LLMs in MLLMs. Experiments on five state-of-the-art (SoTA) MLLMs demonstrate that our ECSO enhances model safety significantly (e.g., a 37.6% improvement on the MM-SafetyBench (SD+OCR), and 71.3% on VLSafe for the LLaVA-1.5-7B), while consistently maintaining utility results on common MLLM benchmarks. Furthermore, we show that ECSO can be used as a data engine to generate supervised-finetuning (SFT) data for MLLM alignment without extra human intervention.



## **5. Risk and Response in Large Language Models: Evaluating Key Threat Categories**

cs.CL

19 pages, 14 figures

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.14988v1) [paper-pdf](http://arxiv.org/pdf/2403.14988v1)

**Authors**: Bahareh Harandizadeh, Abel Salinas, Fred Morstatter

**Abstract**: This paper explores the pressing issue of risk assessment in Large Language Models (LLMs) as they become increasingly prevalent in various applications. Focusing on how reward models, which are designed to fine-tune pretrained LLMs to align with human values, perceive and categorize different types of risks, we delve into the challenges posed by the subjective nature of preference-based training data. By utilizing the Anthropic Red-team dataset, we analyze major risk categories, including Information Hazards, Malicious Uses, and Discrimination/Hateful content. Our findings indicate that LLMs tend to consider Information Hazards less harmful, a finding confirmed by a specially developed regression model. Additionally, our analysis shows that LLMs respond less stringently to Information Hazards compared to other risks. The study further reveals a significant vulnerability of LLMs to jailbreaking attacks in Information Hazard scenarios, highlighting a critical security concern in LLM risk assessment and emphasizing the need for improved AI safety measures.



## **6. BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP**

cs.CV

14 pages, 6 figures

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2311.16194v2) [paper-pdf](http://arxiv.org/pdf/2311.16194v2)

**Authors**: Jiawang Bai, Kuofeng Gao, Shaobo Min, Shu-Tao Xia, Zhifeng Li, Wei Liu

**Abstract**: Contrastive Vision-Language Pre-training, known as CLIP, has shown promising effectiveness in addressing downstream image recognition tasks. However, recent works revealed that the CLIP model can be implanted with a downstream-oriented backdoor. On downstream tasks, one victim model performs well on clean samples but predicts a specific target class whenever a specific trigger is present. For injecting a backdoor, existing attacks depend on a large amount of additional data to maliciously fine-tune the entire pre-trained CLIP model, which makes them inapplicable to data-limited scenarios. In this work, motivated by the recent success of learnable prompts, we address this problem by injecting a backdoor into the CLIP model in the prompt learning stage. Our method named BadCLIP is built on a novel and effective mechanism in backdoor attacks on CLIP, i.e., influencing both the image and text encoders with the trigger. It consists of a learnable trigger applied to images and a trigger-aware context generator, such that the trigger can change text features via trigger-aware prompts, resulting in a powerful and generalizable attack. Extensive experiments conducted on 11 datasets verify that the clean accuracy of BadCLIP is similar to those of advanced prompt learning methods and the attack success rate is higher than 99% in most cases. BadCLIP is also generalizable to unseen classes, and shows a strong generalization capability under cross-dataset and cross-domain settings.



## **7. Unveiling Typographic Deceptions: Insights of the Typographic Vulnerability in Large Vision-Language Model**

cs.CV

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2402.19150v2) [paper-pdf](http://arxiv.org/pdf/2402.19150v2)

**Authors**: Hao Cheng, Erjia Xiao, Jindong Gu, Le Yang, Jinhao Duan, Jize Zhang, Jiahang Cao, Kaidi Xu, Renjing Xu

**Abstract**: Large Vision-Language Models (LVLMs) rely on vision encoders and Large Language Models (LLMs) to exhibit remarkable capabilities on various multi-modal tasks in the joint space of vision and language. However, the Typographic Attack, which disrupts vision-language models (VLMs) such as Contrastive Language-Image Pretraining (CLIP), has also been expected to be a security threat to LVLMs. Firstly, we verify typographic attacks on current well-known commercial and open-source LVLMs and uncover the widespread existence of this threat. Secondly, to better assess this vulnerability, we propose the most comprehensive and largest-scale Typographic Dataset to date. The Typographic Dataset not only considers the evaluation of typographic attacks under various multi-modal tasks but also evaluates the effects of typographic attacks, influenced by texts generated with diverse factors. Based on the evaluation results, we investigate the causes why typographic attacks may impact VLMs and LVLMs, leading to three highly insightful discoveries. By the examination of our discoveries and experimental validation in the Typographic Dataset, we reduce the performance degradation from $42.07\%$ to $13.90\%$ when LVLMs confront typographic attacks.



## **8. Detoxifying Large Language Models via Knowledge Editing**

cs.CL

Ongoing work. Project website:  https://zjunlp.github.io/project/SafeEdit Benchmark:  https://huggingface.co/datasets/zjunlp/SafeEdit Code:  https://github.com/zjunlp/EasyEdit

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14472v1) [paper-pdf](http://arxiv.org/pdf/2403.14472v1)

**Authors**: Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, Huajun Chen

**Abstract**: This paper investigates using knowledge editing techniques to detoxify Large Language Models (LLMs). We construct a benchmark, SafeEdit, which covers nine unsafe categories with various powerful attack prompts and equips comprehensive metrics for systematic evaluation. We conduct experiments to compare knowledge editing approaches with previous baselines, indicating that knowledge editing has the potential to efficiently detoxify LLMs with limited impact on general performance. Then, we propose a simple yet effective baseline, dubbed Detoxifying with Intraoperative Neural Monitoring (DINM), to diminish the toxicity of LLMs within a few tuning steps via only one instance. We further provide an in-depth analysis of the internal mechanism for various detoxify approaches, demonstrating that previous methods like SFT and DPO may merely suppress the activations of toxic parameters, while DINM mitigates the toxicity of the toxic parameters to a certain extent, making permanent adjustments. We hope that these insights could shed light on future work of developing detoxifying approaches and the underlying knowledge mechanisms of LLMs. Code and benchmark are available at https://github.com/zjunlp/EasyEdit.



## **9. $\nabla τ$: Gradient-based and Task-Agnostic machine Unlearning**

cs.LG

14 pages, 2 figures

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14339v1) [paper-pdf](http://arxiv.org/pdf/2403.14339v1)

**Authors**: Daniel Trippa, Cesare Campagnano, Maria Sofia Bucarelli, Gabriele Tolomei, Fabrizio Silvestri

**Abstract**: Machine Unlearning, the process of selectively eliminating the influence of certain data examples used during a model's training, has gained significant attention as a means for practitioners to comply with recent data protection regulations. However, existing unlearning methods face critical drawbacks, including their prohibitively high cost, often associated with a large number of hyperparameters, and the limitation of forgetting only relatively small data portions. This often makes retraining the model from scratch a quicker and more effective solution. In this study, we introduce Gradient-based and Task-Agnostic machine Unlearning ($\nabla \tau$), an optimization framework designed to remove the influence of a subset of training data efficiently. It applies adaptive gradient ascent to the data to be forgotten while using standard gradient descent for the remaining data. $\nabla \tau$ offers multiple benefits over existing approaches. It enables the unlearning of large sections of the training dataset (up to 30%). It is versatile, supporting various unlearning tasks (such as subset forgetting or class removal) and applicable across different domains (images, text, etc.). Importantly, $\nabla \tau$ requires no hyperparameter adjustments, making it a more appealing option than retraining the model from scratch. We evaluate our framework's effectiveness using a set of well-established Membership Inference Attack metrics, demonstrating up to 10% enhancements in performance compared to state-of-the-art methods without compromising the original model's accuracy.



## **10. Large Language Models for Blockchain Security: A Systematic Literature Review**

cs.CR

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14280v1) [paper-pdf](http://arxiv.org/pdf/2403.14280v1)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.



## **11. FMM-Attack: A Flow-based Multi-modal Adversarial Attack on Video-based LLMs**

cs.CV

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.13507v2) [paper-pdf](http://arxiv.org/pdf/2403.13507v2)

**Authors**: Jinmin Li, Kuofeng Gao, Yang Bai, Jingyun Zhang, Shu-tao Xia, Yisen Wang

**Abstract**: Despite the remarkable performance of video-based large language models (LLMs), their adversarial threat remains unexplored. To fill this gap, we propose the first adversarial attack tailored for video-based LLMs by crafting flow-based multi-modal adversarial perturbations on a small fraction of frames within a video, dubbed FMM-Attack. Extensive experiments show that our attack can effectively induce video-based LLMs to generate incorrect answers when videos are added with imperceptible adversarial perturbations. Intriguingly, our FMM-Attack can also induce garbling in the model output, prompting video-based LLMs to hallucinate. Overall, our observations inspire a further understanding of multi-modal robustness and safety-related feature alignment across different modalities, which is of great importance for various large multi-modal models. Our code is available at https://github.com/THU-Kingmin/FMM-Attack.



## **12. AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models**

cs.CL

Published as a conference paper at ICLR 2024. Code is available at  https://github.com/SheltonLiu-N/AutoDAN

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2310.04451v2) [paper-pdf](http://arxiv.org/pdf/2310.04451v2)

**Authors**: Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao

**Abstract**: The aligned Large Language Models (LLMs) are powerful language understanding and decision-making tools that are created through extensive alignment with human feedback. However, these large models remain susceptible to jailbreak attacks, where adversaries manipulate prompts to elicit malicious outputs that should not be given by aligned LLMs. Investigating jailbreak prompts can lead us to delve into the limitations of LLMs and further guide us to secure them. Unfortunately, existing jailbreak techniques suffer from either (1) scalability issues, where attacks heavily rely on manual crafting of prompts, or (2) stealthiness problems, as attacks depend on token-based algorithms to generate prompts that are often semantically meaningless, making them susceptible to detection through basic perplexity testing. In light of these challenges, we intend to answer this question: Can we develop an approach that can automatically generate stealthy jailbreak prompts? In this paper, we introduce AutoDAN, a novel jailbreak attack against aligned LLMs. AutoDAN can automatically generate stealthy jailbreak prompts by the carefully designed hierarchical genetic algorithm. Extensive evaluations demonstrate that AutoDAN not only automates the process while preserving semantic meaningfulness, but also demonstrates superior attack strength in cross-model transferability, and cross-sample universality compared with the baseline. Moreover, we also compare AutoDAN with perplexity-based defense methods and show that AutoDAN can bypass them effectively.



## **13. A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly**

cs.CR

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2312.02003v3) [paper-pdf](http://arxiv.org/pdf/2312.02003v3)

**Authors**: Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, Yue Zhang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and Bard, have revolutionized natural language understanding and generation. They possess deep language comprehension, human-like text generation capabilities, contextual awareness, and robust problem-solving skills, making them invaluable in various domains (e.g., search engines, customer support, translation). In the meantime, LLMs have also gained traction in the security community, revealing security vulnerabilities and showcasing their potential in security-related tasks. This paper explores the intersection of LLMs with security and privacy. Specifically, we investigate how LLMs positively impact security and privacy, potential risks and threats associated with their use, and inherent vulnerabilities within LLMs. Through a comprehensive literature review, the paper categorizes the papers into "The Good" (beneficial LLM applications), "The Bad" (offensive applications), and "The Ugly" (vulnerabilities of LLMs and their defenses). We have some interesting findings. For example, LLMs have proven to enhance code security (code vulnerability detection) and data privacy (data confidentiality protection), outperforming traditional methods. However, they can also be harnessed for various attacks (particularly user-level attacks) due to their human-like reasoning abilities. We have identified areas that require further research efforts. For example, Research on model and parameter extraction attacks is limited and often theoretical, hindered by LLM parameter scale and confidentiality. Safe instruction tuning, a recent development, requires more exploration. We hope that our work can shed light on the LLMs' potential to both bolster and jeopardize cybersecurity.



## **14. Defending Against Indirect Prompt Injection Attacks With Spotlighting**

cs.CR

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.14720v1) [paper-pdf](http://arxiv.org/pdf/2403.14720v1)

**Authors**: Keegan Hines, Gary Lopez, Matthew Hall, Federico Zarfati, Yonatan Zunger, Emre Kiciman

**Abstract**: Large Language Models (LLMs), while powerful, are built and trained to process a single text input. In common applications, multiple inputs can be processed by concatenating them together into a single stream of text. However, the LLM is unable to distinguish which sections of prompt belong to various input sources. Indirect prompt injection attacks take advantage of this vulnerability by embedding adversarial instructions into untrusted data being processed alongside user commands. Often, the LLM will mistake the adversarial instructions as user commands to be followed, creating a security vulnerability in the larger system. We introduce spotlighting, a family of prompt engineering techniques that can be used to improve LLMs' ability to distinguish among multiple sources of input. The key insight is to utilize transformations of an input to provide a reliable and continuous signal of its provenance. We evaluate spotlighting as a defense against indirect prompt injection attacks, and find that it is a robust defense that has minimal detrimental impact to underlying NLP tasks. Using GPT-family models, we find that spotlighting reduces the attack success rate from greater than {50}\% to below {2}\% in our experiments with minimal impact on task efficacy.



## **15. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

cs.CL

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2401.09002v3) [paper-pdf](http://arxiv.org/pdf/2401.09002v3)

**Authors**: Dong shu, Mingyu Jin, Suiyuan Zhu, Beichen Wang, Zihao Zhou, Chong Zhang, Yongfeng Zhang

**Abstract**: In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation aligns with the baseline's trend while offering a more profound and detailed assessment. We believe that by accurately evaluating the effectiveness of attack prompts in the Jailbreak task, our work lays a solid foundation for assessing a wider array of similar or even more complex tasks in the realm of prompt injection, potentially revolutionizing this field.



## **16. BadEdit: Backdooring large language models by model editing**

cs.CR

ICLR 2024

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13355v1) [paper-pdf](http://arxiv.org/pdf/2403.13355v1)

**Authors**: Yanzhou Li, Tianlin Li, Kangjie Chen, Jian Zhang, Shangqing Liu, Wenhan Wang, Tianwei Zhang, Yang Liu

**Abstract**: Mainstream backdoor attack methods typically demand substantial tuning data for poisoning, limiting their practicality and potentially degrading the overall performance when applied to Large Language Models (LLMs). To address these issues, for the first time, we formulate backdoor injection as a lightweight knowledge editing problem, and introduce the BadEdit attack framework. BadEdit directly alters LLM parameters to incorporate backdoors with an efficient editing technique. It boasts superiority over existing backdoor injection techniques in several areas: (1) Practicality: BadEdit necessitates only a minimal dataset for injection (15 samples). (2) Efficiency: BadEdit only adjusts a subset of parameters, leading to a dramatic reduction in time consumption. (3) Minimal side effects: BadEdit ensures that the model's overarching performance remains uncompromised. (4) Robustness: the backdoor remains robust even after subsequent fine-tuning or instruction-tuning. Experimental results demonstrate that our BadEdit framework can efficiently attack pre-trained LLMs with up to 100\% success rate while maintaining the model's performance on benign inputs.



## **17. Mapping LLM Security Landscapes: A Comprehensive Stakeholder Risk Assessment Proposal**

cs.CR

10 pages, 1 figure, 3 tables

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13309v1) [paper-pdf](http://arxiv.org/pdf/2403.13309v1)

**Authors**: Rahul Pankajakshan, Sumitra Biswal, Yuvaraj Govindarajulu, Gilad Gressel

**Abstract**: The rapid integration of Large Language Models (LLMs) across diverse sectors has marked a transformative era, showcasing remarkable capabilities in text generation and problem-solving tasks. However, this technological advancement is accompanied by significant risks and vulnerabilities. Despite ongoing security enhancements, attackers persistently exploit these weaknesses, casting doubts on the overall trustworthiness of LLMs. Compounding the issue, organisations are deploying LLM-integrated systems without understanding the severity of potential consequences. Existing studies by OWASP and MITRE offer a general overview of threats and vulnerabilities but lack a method for directly and succinctly analysing the risks for security practitioners, developers, and key decision-makers who are working with this novel technology. To address this gap, we propose a risk assessment process using tools like the OWASP risk rating methodology which is used for traditional systems. We conduct scenario analysis to identify potential threat agents and map the dependent system components against vulnerability factors. Through this analysis, we assess the likelihood of a cyberattack. Subsequently, we conduct a thorough impact analysis to derive a comprehensive threat matrix. We also map threats against three key stakeholder groups: developers engaged in model fine-tuning, application developers utilizing third-party APIs, and end users. The proposed threat matrix provides a holistic evaluation of LLM-related risks, enabling stakeholders to make informed decisions for effective mitigation strategies. Our outlined process serves as an actionable and comprehensive tool for security practitioners, offering insights for resource management and enhancing the overall system security.



## **18. Bypassing LLM Watermarks with Color-Aware Substitutions**

cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.14719v1) [paper-pdf](http://arxiv.org/pdf/2403.14719v1)

**Authors**: Qilong Wu, Varun Chandrasekaran

**Abstract**: Watermarking approaches are proposed to identify if text being circulated is human or large language model (LLM) generated. The state-of-the-art watermarking strategy of Kirchenbauer et al. (2023a) biases the LLM to generate specific (``green'') tokens. However, determining the robustness of this watermarking method is an open problem. Existing attack methods fail to evade detection for longer text segments. We overcome this limitation, and propose {\em Self Color Testing-based Substitution (SCTS)}, the first ``color-aware'' attack. SCTS obtains color information by strategically prompting the watermarked LLM and comparing output tokens frequencies. It uses this information to determine token colors, and substitutes green tokens with non-green ones. In our experiments, SCTS successfully evades watermark detection using fewer number of edits than related work. Additionally, we show both theoretically and empirically that SCTS can remove the watermark for arbitrarily long watermarked text.



## **19. Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey**

cs.CL

Accepted to NAACL 2024

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2402.09283v2) [paper-pdf](http://arxiv.org/pdf/2402.09283v2)

**Authors**: Zhichen Dong, Zhanhui Zhou, Chao Yang, Jing Shao, Yu Qiao

**Abstract**: Large Language Models (LLMs) are now commonplace in conversation applications. However, their risks of misuse for generating harmful responses have raised serious societal concerns and spurred recent research on LLM conversation safety. Therefore, in this survey, we provide a comprehensive overview of recent studies, covering three critical aspects of LLM conversation safety: attacks, defenses, and evaluations. Our goal is to provide a structured summary that enhances understanding of LLM conversation safety and encourages further investigation into this important subject. For easy reference, we have categorized all the studies mentioned in this survey according to our taxonomy, available at: https://github.com/niconi19/LLM-conversation-safety.



## **20. Review of Generative AI Methods in Cybersecurity**

cs.CR

40 pages

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.08701v2) [paper-pdf](http://arxiv.org/pdf/2403.08701v2)

**Authors**: Yagmur Yigit, William J Buchanan, Madjid G Tehrani, Leandros Maglaras

**Abstract**: Over the last decade, Artificial Intelligence (AI) has become increasingly popular, especially with the use of chatbots such as ChatGPT, Gemini, and DALL-E. With this rise, large language models (LLMs) and Generative AI (GenAI) have also become more prevalent in everyday use. These advancements strengthen cybersecurity's defensive posture and open up new attack avenues for adversaries as well. This paper provides a comprehensive overview of the current state-of-the-art deployments of GenAI, covering assaults, jailbreaking, and applications of prompt injection and reverse psychology. This paper also provides the various applications of GenAI in cybercrimes, such as automated hacking, phishing emails, social engineering, reverse cryptography, creating attack payloads, and creating malware. GenAI can significantly improve the automation of defensive cyber security processes through strategies such as dataset construction, safe code development, threat intelligence, defensive measures, reporting, and cyberattack detection. In this study, we suggest that future research should focus on developing robust ethical norms and innovative defense mechanisms to address the current issues that GenAI creates and to also further encourage an impartial approach to its future application in cybersecurity. Moreover, we underscore the importance of interdisciplinary approaches further to bridge the gap between scientific developments and ethical considerations.



## **21. RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**

cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.13031v1) [paper-pdf](http://arxiv.org/pdf/2403.13031v1)

**Authors**: Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, Bo Li

**Abstract**: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evaluations demonstrate that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks. The innovative use of constrained optimization and a fusion-based guardrail approach represents a significant step forward in developing more secure and reliable LLMs, setting a new standard for content moderation frameworks in the face of evolving digital threats.



## **22. Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices**

cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12503v1) [paper-pdf](http://arxiv.org/pdf/2403.12503v1)

**Authors**: Sara Abdali, Richard Anarfi, CJ Barberan, Jia He

**Abstract**: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.



## **23. Large language models in 6G security: challenges and opportunities**

cs.CR

29 pages, 2 figures

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12239v1) [paper-pdf](http://arxiv.org/pdf/2403.12239v1)

**Authors**: Tri Nguyen, Huong Nguyen, Ahmad Ijaz, Saeid Sheikhi, Athanasios V. Vasilakos, Panos Kostakos

**Abstract**: The rapid integration of Generative AI (GenAI) and Large Language Models (LLMs) in sectors such as education and healthcare have marked a significant advancement in technology. However, this growth has also led to a largely unexplored aspect: their security vulnerabilities. As the ecosystem that includes both offline and online models, various tools, browser plugins, and third-party applications continues to expand, it significantly widens the attack surface, thereby escalating the potential for security breaches. These expansions in the 6G and beyond landscape provide new avenues for adversaries to manipulate LLMs for malicious purposes. We focus on the security aspects of LLMs from the viewpoint of potential adversaries. We aim to dissect their objectives and methodologies, providing an in-depth analysis of known security weaknesses. This will include the development of a comprehensive threat taxonomy, categorizing various adversary behaviors. Also, our research will concentrate on how LLMs can be integrated into cybersecurity efforts by defense teams, also known as blue teams. We will explore the potential synergy between LLMs and blockchain technology, and how this combination could lead to the development of next-generation, fully autonomous security solutions. This approach aims to establish a unified cybersecurity strategy across the entire computing continuum, enhancing overall digital security infrastructure.



## **24. Shifting the Lens: Detecting Malware in npm Ecosystem with Large Language Models**

cs.CR

13 pages, 1 Figure, 7 tables

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12196v1) [paper-pdf](http://arxiv.org/pdf/2403.12196v1)

**Authors**: Nusrat Zahan, Philipp Burckhardt, Mikola Lysenko, Feross Aboukhadijeh, Laurie Williams

**Abstract**: The Gartner 2022 report predicts that 45% of organizations worldwide will encounter software supply chain attacks by 2025, highlighting the urgency to improve software supply chain security for community and national interests. Current malware detection techniques aid in the manual review process by filtering benign and malware packages, yet such techniques have high false-positive rates and limited automation support. Therefore, malware detection techniques could benefit from advanced, more automated approaches for accurate and minimally false-positive results. The goal of this study is to assist security analysts in identifying malicious packages through the empirical study of large language models (LLMs) to detect potential malware in the npm ecosystem.   We present SocketAI Scanner, a multi-stage decision-maker malware detection workflow using iterative self-refinement and zero-shot-role-play-Chain of Thought (CoT) prompting techniques for ChatGPT. We studied 5,115 npm packages (of which 2,180 are malicious) and performed a baseline comparison of the GPT-3 and GPT-4 models with a static analysis tool. Our findings showed promising results for GPT models with low misclassification alert rates. Our baseline comparison demonstrates a notable improvement over static analysis in precision scores above 25% and F1 scores above 15%. We attained precision and F1 scores of 91% and 94%, respectively, for the GPT-3 model. Overall, GPT-4 demonstrates superior performance in precision (99%) and F1 (97%) scores, while GPT-3 presents a cost-effective balance between performance and expenditure.



## **25. EasyJailbreak: A Unified Framework for Jailbreaking Large Language Models**

cs.CL

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12171v1) [paper-pdf](http://arxiv.org/pdf/2403.12171v1)

**Authors**: Weikang Zhou, Xiao Wang, Limao Xiong, Han Xia, Yingshuang Gu, Mingxu Chai, Fukang Zhu, Caishuang Huang, Shihan Dou, Zhiheng Xi, Rui Zheng, Songyang Gao, Yicheng Zou, Hang Yan, Yifan Le, Ruohui Wang, Lijun Li, Jing Shao, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Jailbreak attacks are crucial for identifying and mitigating the security vulnerabilities of Large Language Models (LLMs). They are designed to bypass safeguards and elicit prohibited outputs. However, due to significant differences among various jailbreak methods, there is no standard implementation framework available for the community, which limits comprehensive security evaluations. This paper introduces EasyJailbreak, a unified framework simplifying the construction and evaluation of jailbreak attacks against LLMs. It builds jailbreak attacks using four components: Selector, Mutator, Constraint, and Evaluator. This modular framework enables researchers to easily construct attacks from combinations of novel and existing components. So far, EasyJailbreak supports 11 distinct jailbreak methods and facilitates the security validation of a broad spectrum of LLMs. Our validation across 10 distinct LLMs reveals a significant vulnerability, with an average breach probability of 60% under various jailbreaking attacks. Notably, even advanced models like GPT-3.5-Turbo and GPT-4 exhibit average Attack Success Rates (ASR) of 57% and 33%, respectively. We have released a wealth of resources for researchers, including a web platform, PyPI published package, screencast video, and experimental outputs.



## **26. Navigation as Attackers Wish? Towards Building Robust Embodied Agents under Federated Learning**

cs.AI

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2211.14769v4) [paper-pdf](http://arxiv.org/pdf/2211.14769v4)

**Authors**: Yunchao Zhang, Zonglin Di, Kaiwen Zhou, Cihang Xie, Xin Eric Wang

**Abstract**: Federated embodied agent learning protects the data privacy of individual visual environments by keeping data locally at each client (the individual environment) during training. However, since the local data is inaccessible to the server under federated learning, attackers may easily poison the training data of the local client to build a backdoor in the agent without notice. Deploying such an agent raises the risk of potential harm to humans, as the attackers may easily navigate and control the agent as they wish via the backdoor. Towards Byzantine-robust federated embodied agent learning, in this paper, we study the attack and defense for the task of vision-and-language navigation (VLN), where the agent is required to follow natural language instructions to navigate indoor environments. First, we introduce a simple but effective attack strategy, Navigation as Wish (NAW), in which the malicious client manipulates local trajectory data to implant a backdoor into the global model. Results on two VLN datasets (R2R and RxR) show that NAW can easily navigate the deployed VLN agent regardless of the language instruction, without affecting its performance on normal test sets. Then, we propose a new Prompt-Based Aggregation (PBA) to defend against the NAW attack in federated VLN, which provides the server with a ''prompt'' of the vision-and-language alignment variance between the benign and malicious clients so that they can be distinguished during training. We validate the effectiveness of the PBA method on protecting the global model from the NAW attack, which outperforms other state-of-the-art defense methods by a large margin in the defense metrics on R2R and RxR.



## **27. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

cs.CR

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2312.00029v2) [paper-pdf](http://arxiv.org/pdf/2312.00029v2)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. These attacks can trick seemingly aligned models into giving manufacturing instructions for dangerous materials, inciting violence, or recommending other immoral acts. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM emulating the conscience of a protected, primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis shows that, by using Bergeron to complement models with existing alignment training, we can improve the robustness and safety of multiple, commonly used commercial and open-source LLMs.



## **28. Beyond Gradient and Priors in Privacy Attacks: Leveraging Pooler Layer Inputs of Language Models in Federated Learning**

cs.LG

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2312.05720v4) [paper-pdf](http://arxiv.org/pdf/2312.05720v4)

**Authors**: Jianwei Li, Sheng Liu, Qi Lei

**Abstract**: Language models trained via federated learning (FL) demonstrate impressive capabilities in handling complex tasks while protecting user privacy. Recent studies indicate that leveraging gradient information and prior knowledge can potentially reveal training samples within FL setting. However, these investigations have overlooked the potential privacy risks tied to the intrinsic architecture of the models. This paper presents a two-stage privacy attack strategy that targets the vulnerabilities in the architecture of contemporary language models, significantly enhancing attack performance by initially recovering certain feature directions as additional supervisory signals. Our comparative experiments demonstrate superior attack performance across various datasets and scenarios, highlighting the privacy leakage risk associated with the increasingly complex architectures of language models. We call for the community to recognize and address these potential privacy risks in designing large language models.



## **29. Logits of API-Protected LLMs Leak Proprietary Information**

cs.CL

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.09539v2) [paper-pdf](http://arxiv.org/pdf/2403.09539v2)

**Authors**: Matthew Finlayson, Xiang Ren, Swabha Swayamdipta

**Abstract**: The commercialization of large language models (LLMs) has led to the common practice of high-level API-only access to proprietary models. In this work, we show that even with a conservative assumption about the model architecture, it is possible to learn a surprisingly large amount of non-public information about an API-protected LLM from a relatively small number of API queries (e.g., costing under $1,000 for OpenAI's gpt-3.5-turbo). Our findings are centered on one key observation: most modern LLMs suffer from a softmax bottleneck, which restricts the model outputs to a linear subspace of the full output space. We show that this lends itself to a model image or a model signature which unlocks several capabilities with affordable cost: efficiently discovering the LLM's hidden size, obtaining full-vocabulary outputs, detecting and disambiguating different model updates, identifying the source LLM given a single full LLM output, and even estimating the output layer parameters. Our empirical investigations show the effectiveness of our methods, which allow us to estimate the embedding size of OpenAI's gpt-3.5-turbo to be about 4,096. Lastly, we discuss ways that LLM providers can guard against these attacks, as well as how these capabilities can be viewed as a feature (rather than a bug) by allowing for greater transparency and accountability.



## **30. Scaling Behavior of Machine Translation with Large Language Models under Prompt Injection Attacks**

cs.CL

15 pages, 18 figures, First Workshop on the Scaling Behavior of Large  Language Models (SCALE-LLM 2024)

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09832v1) [paper-pdf](http://arxiv.org/pdf/2403.09832v1)

**Authors**: Zhifan Sun, Antonio Valerio Miceli-Barone

**Abstract**: Large Language Models (LLMs) are increasingly becoming the preferred foundation platforms for many Natural Language Processing tasks such as Machine Translation, owing to their quality often comparable to or better than task-specific models, and the simplicity of specifying the task through natural language instructions or in-context examples. Their generality, however, opens them up to subversion by end users who may embed into their requests instructions that cause the model to behave in unauthorized and possibly unsafe ways. In this work we study these Prompt Injection Attacks (PIAs) on multiple families of LLMs on a Machine Translation task, focusing on the effects of model size on the attack success rates. We introduce a new benchmark data set and we discover that on multiple language pairs and injected prompts written in English, larger models under certain conditions may become more susceptible to successful attacks, an instance of the Inverse Scaling phenomenon (McKenzie et al., 2023). To our knowledge, this is the first work to study non-trivial LLM scaling behaviour in a multi-lingual setting.



## **31. Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models**

cs.CV

Work in progress

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09792v1) [paper-pdf](http://arxiv.org/pdf/2403.09792v1)

**Authors**: Yifan Li, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, Ji-Rong Wen

**Abstract**: In this paper, we study the harmlessness alignment problem of multimodal large language models~(MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method named HADES, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images. Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate~(ASR) of 90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision. Our code and data will be publicly released.



## **32. AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting**

cs.CR

Multimodal Large Language Models Defense, 25 Pages

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09513v1) [paper-pdf](http://arxiv.org/pdf/2403.09513v1)

**Authors**: Yu Wang, Xiaogeng Liu, Yu Li, Muhao Chen, Chaowei Xiao

**Abstract**: With the advent and widespread deployment of Multimodal Large Language Models (MLLMs), the imperative to ensure their safety has become increasingly pronounced. However, with the integration of additional modalities, MLLMs are exposed to new vulnerabilities, rendering them prone to structured-based jailbreak attacks, where semantic content (e.g., "harmful text") has been injected into the images to mislead MLLMs. In this work, we aim to defend against such threats. Specifically, we propose \textbf{Ada}ptive \textbf{Shield} Prompting (\textbf{AdaShield}), which prepends inputs with defense prompts to defend MLLMs against structure-based jailbreak attacks without fine-tuning MLLMs or training additional modules (e.g., post-stage content detector). Initially, we present a manually designed static defense prompt, which thoroughly examines the image and instruction content step by step and specifies response methods to malicious queries. Furthermore, we introduce an adaptive auto-refinement framework, consisting of a target MLLM and a LLM-based defense prompt generator (Defender). These components collaboratively and iteratively communicate to generate a defense prompt. Extensive experiments on the popular structure-based jailbreak attacks and benign datasets show that our methods can consistently improve MLLMs' robustness against structure-based jailbreak attacks without compromising the model's general capabilities evaluated on standard benign tasks. Our code is available at https://github.com/rain305f/AdaShield.



## **33. On Protecting the Data Privacy of Large Language Models (LLMs): A Survey**

cs.CR

18 pages, 4 figures

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.05156v2) [paper-pdf](http://arxiv.org/pdf/2403.05156v2)

**Authors**: Biwei Yan, Kun Li, Minghui Xu, Yueyan Dong, Yue Zhang, Zhaochun Ren, Xiuzhen Cheng

**Abstract**: Large language models (LLMs) are complex artificial intelligence systems capable of understanding, generating and translating human language. They learn language patterns by analyzing large amounts of text data, allowing them to perform writing, conversation, summarizing and other language tasks. When LLMs process and generate large amounts of data, there is a risk of leaking sensitive information, which may threaten data privacy. This paper concentrates on elucidating the data privacy concerns associated with LLMs to foster a comprehensive understanding. Specifically, a thorough investigation is undertaken to delineate the spectrum of data privacy threats, encompassing both passive privacy leakage and active privacy attacks within LLMs. Subsequently, we conduct an assessment of the privacy protection mechanisms employed by LLMs at various stages, followed by a detailed examination of their efficacy and constraints. Finally, the discourse extends to delineate the challenges encountered and outline prospective directions for advancement in the realm of LLM privacy protection.



## **34. AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Adversarial Visual-Instructions**

cs.CV

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09346v1) [paper-pdf](http://arxiv.org/pdf/2403.09346v1)

**Authors**: Hao Zhang, Wenqi Shao, Hong Liu, Yongqiang Ma, Ping Luo, Yu Qiao, Kaipeng Zhang

**Abstract**: Large Vision-Language Models (LVLMs) have shown significant progress in well responding to visual-instructions from users. However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited. To bridge this gap, we introduce AVIBench, a framework designed to analyze the robustness of LVLMs when facing various adversarial visual-instructions (AVIs), including four types of image-based AVIs, ten types of text-based AVIs, and nine types of content bias AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 260K AVIs encompassing five categories of multimodal capabilities (nine tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. AVIBench also serves as a convenient tool for practitioners to evaluate the robustness of LVLMs against AVIs. Our findings and extensive experimental results shed light on the vulnerabilities of LVLMs, and highlight that inherent biases exist even in advanced closed-source LVLMs like GeminiProVision and GPT-4V. This underscores the importance of enhancing the robustness, security, and fairness of LVLMs. The source code and benchmark will be made publicly available.



## **35. What Was Your Prompt? A Remote Keylogging Attack on AI Assistants**

cs.CR

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09751v1) [paper-pdf](http://arxiv.org/pdf/2403.09751v1)

**Authors**: Roy Weiss, Daniel Ayzenshteyn, Guy Amit, Yisroel Mirsky

**Abstract**: AI assistants are becoming an integral part of society, used for asking advice or help in personal and confidential issues. In this paper, we unveil a novel side-channel that can be used to read encrypted responses from AI Assistants over the web: the token-length side-channel. We found that many vendors, including OpenAI and Microsoft, have this side-channel.   However, inferring the content of a response from a token-length sequence alone proves challenging. This is because tokens are akin to words, and responses can be several sentences long leading to millions of grammatically correct sentences. In this paper, we show how this can be overcome by (1) utilizing the power of a large language model (LLM) to translate these sequences, (2) providing the LLM with inter-sentence context to narrow the search space and (3) performing a known-plaintext attack by fine-tuning the model on the target model's writing style.   Using these methods, we were able to accurately reconstruct 29\% of an AI assistant's responses and successfully infer the topic from 55\% of them. To demonstrate the threat, we performed the attack on OpenAI's ChatGPT-4 and Microsoft's Copilot on both browser and API traffic.



## **36. The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?**

cs.CV

Under review. Project page:  https://github.com/Qinyu-Allen-Zhao/LVLM-LP

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09037v1) [paper-pdf](http://arxiv.org/pdf/2403.09037v1)

**Authors**: Qinyu Zhao, Ming Xu, Kartik Gupta, Akshay Asthana, Liang Zheng, Stephen Gould

**Abstract**: Large vision-language models (LVLMs), designed to interpret and respond to human instructions, occasionally generate hallucinated or harmful content due to inappropriate instructions. This study uses linear probing to shed light on the hidden knowledge at the output layer of LVLMs. We demonstrate that the logit distributions of the first tokens contain sufficient information to determine whether to respond to the instructions, including recognizing unanswerable visual questions, defending against multi-modal jailbreaking attack, and identifying deceptive questions. Such hidden knowledge is gradually lost in logits of subsequent tokens during response generation. Then, we illustrate a simple decoding strategy at the generation of the first token, effectively improving the generated content. In experiments, we find a few interesting insights: First, the CLIP model already contains a strong signal for solving these tasks, indicating potential bias in the existing datasets. Second, we observe performance improvement by utilizing the first logit distributions on three additional tasks, including indicting uncertainty in math solving, mitigating hallucination, and image classification. Last, with the same training data, simply finetuning LVLMs improve models' performance but is still inferior to linear probing on these tasks.



## **37. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2312.03853v2) [paper-pdf](http://arxiv.org/pdf/2312.03853v2)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Only a year ago, we witnessed a rise in the use of Large Language Models (LLMs), especially when combined with applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with opposite characteristics as those of the truthful assistants they are supposed to be. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversation followed a role-play style to get the response the assistant was not allowed to provide. By making use of personas, we show that the response that is prohibited is actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. We also introduce several ways of activating such adversarial personas, altogether showing that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.



## **38. SoK: Reducing the Vulnerability of Fine-tuned Language Models to Membership Inference Attacks**

cs.LG

preliminary version

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08481v1) [paper-pdf](http://arxiv.org/pdf/2403.08481v1)

**Authors**: Guy Amit, Abigail Goldsteen, Ariel Farkash

**Abstract**: Natural language processing models have experienced a significant upsurge in recent years, with numerous applications being built upon them. Many of these applications require fine-tuning generic base models on customized, proprietary datasets. This fine-tuning data is especially likely to contain personal or sensitive information about individuals, resulting in increased privacy risk. Membership inference attacks are the most commonly employed attack to assess the privacy leakage of a machine learning model. However, limited research is available on the factors that affect the vulnerability of language models to this kind of attack, or on the applicability of different defense strategies in the language domain. We provide the first systematic review of the vulnerability of fine-tuned large language models to membership inference attacks, the various factors that come into play, and the effectiveness of different defense strategies. We find that some training methods provide significantly reduced privacy risk, with the combination of differential privacy and low-rank adaptors achieving the best privacy protection against these attacks.



## **39. The Philosopher's Stone: Trojaning Plugins of Large Language Models**

cs.CR

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2312.00374v2) [paper-pdf](http://arxiv.org/pdf/2312.00374v2)

**Authors**: Tian Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Shaofeng Li, Yan Meng, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses LLM-enhanced paraphrasing to polish benchmark poisoned datasets. In contrast, in the absence of a dataset, FUSION leverages an over-poisoning procedure to transform a benign adaptor. In our experiments, we first conduct two case studies to demonstrate that a compromised LLM agent can execute malware to control system (e.g., LLM-driven robot) or launch a spear-phishing attack. Then, in terms of targeted misinformation, we show that our attacks provide higher attack effectiveness than the baseline and, for the purpose of attracting downloads, preserve or improve the adapter's utility. Finally, we design and evaluate three potential defenses, yet none proved entirely effective in safeguarding against our attacks.



## **40. Tastle: Distract Large Language Models for Automatic Jailbreak Attack**

cs.CR

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08424v1) [paper-pdf](http://arxiv.org/pdf/2403.08424v1)

**Authors**: Zeguan Xiao, Yan Yang, Guanhua Chen, Yun Chen

**Abstract**: Large language models (LLMs) have achieved significant advances in recent days. Extensive efforts have been made before the public release of LLMs to align their behaviors with human values. The primary goal of alignment is to ensure their helpfulness, honesty and harmlessness. However, even meticulously aligned LLMs remain vulnerable to malicious manipulations such as jailbreaking, leading to unintended behaviors. The jailbreak is to intentionally develop a malicious prompt that escapes from the LLM security restrictions to produce uncensored detrimental contents. Previous works explore different jailbreak methods for red teaming LLMs, yet they encounter challenges regarding to effectiveness and scalability. In this work, we propose Tastle, a novel black-box jailbreak framework for automated red teaming of LLMs. We designed malicious content concealing and memory reframing with an iterative optimization algorithm to jailbreak LLMs, motivated by the research about the distractibility and over-confidence phenomenon of LLMs. Extensive experiments of jailbreaking both open-source and proprietary LLMs demonstrate the superiority of our framework in terms of effectiveness, scalability and transferability. We also evaluate the effectiveness of existing jailbreak defense methods against our attack and highlight the crucial need to develop more effective and practical defense strategies.



## **41. Duwak: Dual Watermarks in Large Language Models**

cs.LG

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.13000v1) [paper-pdf](http://arxiv.org/pdf/2403.13000v1)

**Authors**: Chaoyi Zhu, Jeroen Galjaard, Pin-Yu Chen, Lydia Y. Chen

**Abstract**: As large language models (LLM) are increasingly used for text generation tasks, it is critical to audit their usages, govern their applications, and mitigate their potential harms. Existing watermark techniques are shown effective in embedding single human-imperceptible and machine-detectable patterns without significantly affecting generated text quality and semantics. However, the efficiency in detecting watermarks, i.e., the minimum number of tokens required to assert detection with significance and robustness against post-editing, is still debatable. In this paper, we propose, Duwak, to fundamentally enhance the efficiency and quality of watermarking by embedding dual secret patterns in both token probability distribution and sampling schemes. To mitigate expression degradation caused by biasing toward certain tokens, we design a contrastive search to watermark the sampling scheme, which minimizes the token repetition and enhances the diversity. We theoretically explain the interdependency of the two watermarks within Duwak. We evaluate Duwak extensively on Llama2 under various post-editing attacks, against four state-of-the-art watermarking techniques and combinations of them. Our results show that Duwak marked text achieves the highest watermarked text quality at the lowest required token count for detection, up to 70% tokens less than existing approaches, especially under post paraphrasing.



## **42. MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2311.17600v2) [paper-pdf](http://arxiv.org/pdf/2311.17600v2)

**Authors**: Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Multimodal Large Language Models (MLLMs) remains understudied. In this paper, we observe that Multimodal Large Language Models (MLLMs) can be easily compromised by query-relevant images, as if the text query itself were malicious. To address this, we introduce MM-SafetyBench, a comprehensive framework designed for conducting safety-critical evaluations of MLLMs against such image-based manipulations. We have compiled a dataset comprising 13 scenarios, resulting in a total of 5,040 text-image pairs. Our analysis across 12 state-of-the-art models reveals that MLLMs are susceptible to breaches instigated by our approach, even when the equipped LLMs have been safety-aligned. In response, we propose a straightforward yet effective prompting strategy to enhance the resilience of MLLMs against these types of attacks. Our work underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source MLLMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **43. Poisoning Programs by Un-Repairing Code: Security Concerns of AI-generated Code**

cs.CR

Accepted at The 1st IEEE International Workshop on Reliable and  Secure AI for Software Engineering (ReSAISE), co-located with ISSRE 2023

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06675v1) [paper-pdf](http://arxiv.org/pdf/2403.06675v1)

**Authors**: Cristina Improta

**Abstract**: AI-based code generators have gained a fundamental role in assisting developers in writing software starting from natural language (NL). However, since these large language models are trained on massive volumes of data collected from unreliable online sources (e.g., GitHub, Hugging Face), AI models become an easy target for data poisoning attacks, in which an attacker corrupts the training data by injecting a small amount of poison into it, i.e., astutely crafted malicious samples. In this position paper, we address the security of AI code generators by identifying a novel data poisoning attack that results in the generation of vulnerable code. Next, we devise an extensive evaluation of how these attacks impact state-of-the-art models for code generation. Lastly, we discuss potential solutions to overcome this threat.



## **44. FedPIT: Towards Privacy-preserving and Few-shot Federated Instruction Tuning**

cs.CR

Work in process

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2403.06131v1) [paper-pdf](http://arxiv.org/pdf/2403.06131v1)

**Authors**: Zhuo Zhang, Jingyuan Zhang, Jintao Huang, Lizhen Qu, Hongzhi Zhang, Zenglin Xu

**Abstract**: Instruction tuning has proven essential for enhancing the performance of large language models (LLMs) in generating human-aligned responses. However, collecting diverse, high-quality instruction data for tuning poses challenges, particularly in privacy-sensitive domains. Federated instruction tuning (FedIT) has emerged as a solution, leveraging federated learning from multiple data owners while preserving privacy. Yet, it faces challenges due to limited instruction data and vulnerabilities to training data extraction attacks. To address these issues, we propose a novel federated algorithm, FedPIT, which utilizes LLMs' in-context learning capability to self-generate task-specific synthetic data for training autonomously. Our method employs parameter-isolated training to maintain global parameters trained on synthetic data and local parameters trained on augmented local data, effectively thwarting data extraction attacks. Extensive experiments on real-world medical data demonstrate the effectiveness of FedPIT in improving federated few-shot performance while preserving privacy and robustness against data heterogeneity.



## **45. Language-Driven Anchors for Zero-Shot Adversarial Robustness**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2301.13096v3) [paper-pdf](http://arxiv.org/pdf/2301.13096v3)

**Authors**: Xiao Li, Wei Zhang, Yining Liu, Zhanhao Hu, Bo Zhang, Xiaolin Hu

**Abstract**: Deep Neural Networks (DNNs) are known to be susceptible to adversarial attacks. Previous researches mainly focus on improving adversarial robustness in the fully supervised setting, leaving the challenging domain of zero-shot adversarial robustness an open question. In this work, we investigate this domain by leveraging the recent advances in large vision-language models, such as CLIP, to introduce zero-shot adversarial robustness to DNNs. We propose LAAT, a Language-driven, Anchor-based Adversarial Training strategy. LAAT utilizes the features of a text encoder for each category as fixed anchors (normalized feature embeddings) for each category, which are then employed for adversarial training. By leveraging the semantic consistency of the text encoders, LAAT aims to enhance the adversarial robustness of the image model on novel categories. However, naively using text encoders leads to poor results. Through analysis, we identified the issue to be the high cosine similarity between text encoders. We then design an expansion algorithm and an alignment cross-entropy loss to alleviate the problem. Our experimental results demonstrated that LAAT significantly improves zero-shot adversarial robustness over state-of-the-art methods. LAAT has the potential to enhance adversarial robustness by large-scale multimodal models, especially when labeled data is unavailable during training.



## **46. From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude**

cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2310.19181v2) [paper-pdf](http://arxiv.org/pdf/2310.19181v2)

**Authors**: Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The advanced capabilities of Large Language Models (LLMs) have made them invaluable across various applications, from conversational agents and content creation to data analysis, research, and innovation. However, their effectiveness and accessibility also render them susceptible to abuse for generating malicious content, including phishing attacks. This study explores the potential of using four popular commercially available LLMs, i.e., ChatGPT (GPT 3.5 Turbo), GPT 4, Claude, and Bard, to generate functional phishing attacks using a series of malicious prompts. We discover that these LLMs can generate both phishing websites and emails that can convincingly imitate well-known brands and also deploy a range of evasive tactics that are used to elude detection mechanisms employed by anti-phishing systems. These attacks can be generated using unmodified or "vanilla" versions of these LLMs without requiring any prior adversarial exploits such as jailbreaking. We evaluate the performance of the LLMs towards generating these attacks and find that they can also be utilized to create malicious prompts that, in turn, can be fed back to the model to generate phishing scams - thus massively reducing the prompt-engineering effort required by attackers to scale these threats. As a countermeasure, we build a BERT-based automated detection tool that can be used for the early detection of malicious prompts to prevent LLMs from generating phishing content. Our model is transferable across all four commercial LLMs, attaining an average accuracy of 96% for phishing website prompts and 94% for phishing email prompts. We also disclose the vulnerabilities to the concerned LLMs, with Google acknowledging it as a severe issue. Our detection model is available for use at Hugging Face, as well as a ChatGPT Actions plugin.



## **47. Can LLMs Follow Simple Rules?**

cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules;  revised content

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2311.04235v3) [paper-pdf](http://arxiv.org/pdf/2311.04235v3)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Basel Alomair, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Existing evaluations of adversarial attacks and defenses on LLMs generally require either expensive manual review or unreliable heuristic checks. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 14 simple text scenarios in which the model is instructed to obey various rules while interacting with the user. Each scenario has a programmatic evaluation function to determine whether the model has broken any rules in a conversation. Our evaluations of proprietary and open models show that almost all current models struggle to follow scenario rules, even on straightforward test cases. We also demonstrate that simple optimization attacks suffice to significantly increase failure rates on test cases. We conclude by exploring two potential avenues for improvement: test-time steering and supervised fine-tuning.



## **48. Warfare:Breaking the Watermark Protection of AI-Generated Content**

cs.CV

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2310.07726v3) [paper-pdf](http://arxiv.org/pdf/2310.07726v3)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.



## **49. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

cs.CL

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2312.14197v3) [paper-pdf](http://arxiv.org/pdf/2312.14197v3)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: The integration of large language models (LLMs) with external content has enabled more up-to-date and wide-ranging applications of LLMs, such as Microsoft Copilot. However, this integration has also exposed LLMs to the risk of indirect prompt injection attacks, where an attacker can embed malicious instructions within external content, compromising LLM output and causing responses to deviate from user expectations. To investigate this important but underexplored issue, we introduce the first benchmark for indirect prompt injection attacks, named BIPIA, to evaluate the risk of such attacks. Based on the evaluation, our work makes a key analysis of the underlying reason for the success of the attack, namely the inability of LLMs to distinguish between instructions and external content and the absence of LLMs' awareness to not execute instructions within external content. Building upon this analysis, we develop two black-box methods based on prompt learning and a white-box defense method based on fine-tuning with adversarial training accordingly. Experimental results demonstrate that black-box defenses are highly effective in mitigating these attacks, while the white-box defense reduces the attack success rate to near-zero levels. Overall, our work systematically investigates indirect prompt injection attacks by introducing a benchmark, analyzing the underlying reason for the success of the attack, and developing an initial set of defenses.



## **50. SecGPT: An Execution Isolation Architecture for LLM-Based Systems**

cs.CR

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.04960v1) [paper-pdf](http://arxiv.org/pdf/2403.04960v1)

**Authors**: Yuhao Wu, Franziska Roesner, Tadayoshi Kohno, Ning Zhang, Umar Iqbal

**Abstract**: Large language models (LLMs) extended as systems, such as ChatGPT, have begun supporting third-party applications. These LLM apps leverage the de facto natural language-based automated execution paradigm of LLMs: that is, apps and their interactions are defined in natural language, provided access to user data, and allowed to freely interact with each other and the system. These LLM app ecosystems resemble the settings of earlier computing platforms, where there was insufficient isolation between apps and the system. Because third-party apps may not be trustworthy, and exacerbated by the imprecision of the natural language interfaces, the current designs pose security and privacy risks for users. In this paper, we propose SecGPT, an architecture for LLM-based systems that aims to mitigate the security and privacy issues that arise with the execution of third-party apps. SecGPT's key idea is to isolate the execution of apps and more precisely mediate their interactions outside of their isolated environments. We evaluate SecGPT against a number of case study attacks and demonstrate that it protects against many security, privacy, and safety issues that exist in non-isolated LLM-based systems. The performance overhead incurred by SecGPT to improve security is under 0.3x for three-quarters of the tested queries. To foster follow-up research, we release SecGPT's source code at https://github.com/llm-platform-security/SecGPT.



