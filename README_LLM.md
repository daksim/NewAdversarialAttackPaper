# Latest Large Language Model Attack Papers
**update at 2024-03-04 16:53:26**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

cs.CR

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2402.16914v2) [paper-pdf](http://arxiv.org/pdf/2402.16914v2)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.



## **2. Here's a Free Lunch: Sanitizing Backdoored Models with Model Merge**

cs.CL

work in progress

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19334v1) [paper-pdf](http://arxiv.org/pdf/2402.19334v1)

**Authors**: Ansh Arora, Xuanli He, Maximilian Mozes, Srinibas Swain, Mark Dras, Qiongkai Xu

**Abstract**: The democratization of pre-trained language models through open-source initiatives has rapidly advanced innovation and expanded access to cutting-edge technologies. However, this openness also brings significant security risks, including backdoor attacks, where hidden malicious behaviors are triggered by specific inputs, compromising natural language processing (NLP) system integrity and reliability. This paper suggests that merging a backdoored model with other homogeneous models can remediate backdoor vulnerabilities even if such models are not entirely secure. In our experiments, we explore various models (BERT-Base, RoBERTa-Large, Llama2-7B, and Mistral-7B) and datasets (SST-2, OLID, AG News, and QNLI). Compared to multiple advanced defensive approaches, our method offers an effective and efficient inference-stage defense against backdoor attacks without additional resources or specific knowledge. Our approach consistently outperforms the other advanced baselines, leading to an average of 75% reduction in the attack success rate. Since model merging has been an established approach for improving model performance, the extra advantage it provides regarding defense can be seen as a cost-free bonus.



## **3. PRSA: Prompt Reverse Stealing Attacks against Large Language Models**

cs.CR

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19200v1) [paper-pdf](http://arxiv.org/pdf/2402.19200v1)

**Authors**: Yong Yang, Xuhong Zhang, Yi Jiang, Xi Chen, Haoyu Wang, Shouling Ji, Zonghui Wang

**Abstract**: Prompt, recognized as crucial intellectual property, enables large language models (LLMs) to perform specific tasks without the need of fine-tuning, underscoring their escalating importance. With the rise of prompt-based services, such as prompt marketplaces and LLM applications, providers often display prompts' capabilities through input-output examples to attract users. However, this paradigm raises a pivotal security concern: does the exposure of input-output pairs pose the risk of potential prompt leakage, infringing on the intellectual property rights of the developers? To our knowledge, this problem still has not been comprehensively explored yet. To remedy this gap, in this paper, we perform the first in depth exploration and propose a novel attack framework for reverse-stealing prompts against commercial LLMs, namely PRSA. The main idea of PRSA is that by analyzing the critical features of the input-output pairs, we mimic and gradually infer (steal) the target prompts. In detail, PRSA mainly consists of two key phases: prompt mutation and prompt pruning. In the mutation phase, we propose a prompt attention algorithm based on differential feedback to capture these critical features for effectively inferring the target prompts. In the prompt pruning phase, we identify and mask the words dependent on specific inputs, enabling the prompts to accommodate diverse inputs for generalization. Through extensive evaluation, we verify that PRSA poses a severe threat in real world scenarios. We have reported these findings to prompt service providers and actively collaborate with them to take protective measures for prompt copyright.



## **4. A Semantic Invariant Robust Watermark for Large Language Models**

cs.CR

ICLR2024, 21 pages, 10 figures, 6 tables

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2310.06356v2) [paper-pdf](http://arxiv.org/pdf/2310.06356v2)

**Authors**: Aiwei Liu, Leyi Pan, Xuming Hu, Shiao Meng, Lijie Wen

**Abstract**: Watermark algorithms for large language models (LLMs) have achieved extremely high accuracy in detecting text generated by LLMs. Such algorithms typically involve adding extra watermark logits to the LLM's logits at each generation step. However, prior algorithms face a trade-off between attack robustness and security robustness. This is because the watermark logits for a token are determined by a certain number of preceding tokens; a small number leads to low security robustness, while a large number results in insufficient attack robustness. In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack robustness and security robustness. The watermark logits in our work are determined by the semantics of all preceding tokens. Specifically, we utilize another embedding LLM to generate semantic embeddings for all preceding tokens, and then these semantic embeddings are transformed into the watermark logits through our trained watermark model. Subsequent analyses and experiments demonstrated the attack robustness of our method in semantically invariant settings: synonym substitution and text paraphrasing settings. Finally, we also show that our watermark possesses adequate security robustness. Our code and data are available at https://github.com/THU-BPM/Robust_Watermark.



## **5. Typographic Attacks in Large Multimodal Models Can be Alleviated by More Informative Prompts**

cs.CV

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19150v1) [paper-pdf](http://arxiv.org/pdf/2402.19150v1)

**Authors**: Hao Cheng, Erjia Xiao, Renjing Xu

**Abstract**: Large Multimodal Models (LMMs) rely on pre-trained Vision Language Models (VLMs) and Large Language Models (LLMs) to perform amazing emergent abilities on various multimodal tasks in the joint space of vision and language. However, the Typographic Attack, which shows disruption to VLMs, has also been certified as a security vulnerability to LMMs. In this work, we first comprehensively investigate the distractibility of LMMs by typography. In particular, we introduce the Typographic Dataset designed to evaluate distractibility across various multi-modal subtasks, such as object recognition, visual attributes detection, enumeration, arithmetic computation, and commonsense reasoning. To further study the effect of typographic patterns on performance, we also scrutinize the effect of tuning various typographic factors, encompassing font size, color, opacity, and spatial positioning of typos. We discover that LMMs can partially distinguish visual contents and typos when confronting typographic attacks, which suggests that embeddings from vision encoders contain enough information to distinguish visual contents and typos in images. Inspired by such phenomena, we demonstrate that CLIP's performance of zero-shot classification on typo-ridden images can be significantly improved by providing more informative texts to match images. Furthermore, we also prove that LMMs can utilize more informative prompts to leverage information in embeddings to differentiate between visual content and typos. Finally, we propose a prompt information enhancement method that can effectively mitigate the effects of typography.



## **6. Cognitive Overload: Jailbreaking Large Language Models with Overloaded Logical Thinking**

cs.CL

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2311.09827v2) [paper-pdf](http://arxiv.org/pdf/2311.09827v2)

**Authors**: Nan Xu, Fei Wang, Ben Zhou, Bang Zheng Li, Chaowei Xiao, Muhao Chen

**Abstract**: While large language models (LLMs) have demonstrated increasing power, they have also given rise to a wide range of harmful behaviors. As representatives, jailbreak attacks can provoke harmful or unethical responses from LLMs, even after safety alignment. In this paper, we investigate a novel category of jailbreak attacks specifically designed to target the cognitive structure and processes of LLMs. Specifically, we analyze the safety vulnerability of LLMs in the face of (1) multilingual cognitive overload, (2) veiled expression, and (3) effect-to-cause reasoning. Different from previous jailbreak attacks, our proposed cognitive overload is a black-box attack with no need for knowledge of model architecture or access to model weights. Experiments conducted on AdvBench and MasterKey reveal that various LLMs, including both popular open-source model Llama 2 and the proprietary model ChatGPT, can be compromised through cognitive overload. Motivated by cognitive psychology work on managing cognitive load, we further investigate defending cognitive overload attack from two perspectives. Empirical studies show that our cognitive overload from three perspectives can jailbreak all studied LLMs successfully, while existing defense strategies can hardly mitigate the caused malicious uses effectively.



## **7. Vaccine: Perturbation-aware Alignment for Large Language Model**

cs.LG

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.01109v3) [paper-pdf](http://arxiv.org/pdf/2402.01109v3)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.



## **8. Defending Large Language Models against Jailbreak Attacks via Semantic Smoothing**

cs.CL

37 pages

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.16192v2) [paper-pdf](http://arxiv.org/pdf/2402.16192v2)

**Authors**: Jiabao Ji, Bairu Hou, Alexander Robey, George J. Pappas, Hamed Hassani, Yang Zhang, Eric Wong, Shiyu Chang

**Abstract**: Aligned large language models (LLMs) are vulnerable to jailbreaking attacks, which bypass the safeguards of targeted LLMs and fool them into generating objectionable content. While initial defenses show promise against token-based threat models, there do not exist defenses that provide robustness against semantic attacks and avoid unfavorable trade-offs between robustness and nominal performance. To meet this need, we propose SEMANTICSMOOTH, a smoothing-based defense that aggregates the predictions of multiple semantically transformed copies of a given input prompt. Experimental results demonstrate that SEMANTICSMOOTH achieves state-of-the-art robustness against GCG, PAIR, and AutoDAN attacks while maintaining strong nominal performance on instruction following benchmarks such as InstructionFollowing and AlpacaEval. The codes will be publicly available at https://github.com/UCSB-NLP-Chang/SemanticSmooth.



## **9. Defending LLMs against Jailbreaking Attacks via Backtranslation**

cs.CL

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.16459v2) [paper-pdf](http://arxiv.org/pdf/2402.16459v2)

**Authors**: Yihan Wang, Zhouxing Shi, Andrew Bai, Cho-Jui Hsieh

**Abstract**: Although many large language models (LLMs) have been trained to refuse harmful requests, they are still vulnerable to jailbreaking attacks, which rewrite the original prompt to conceal its harmful intent. In this paper, we propose a new method for defending LLMs against jailbreaking attacks by ``backtranslation''. Specifically, given an initial response generated by the target LLM from an input prompt, our backtranslation prompts a language model to infer an input prompt that can lead to the response. The inferred prompt is called the backtranslated prompt which tends to reveal the actual intent of the original prompt, since it is generated based on the LLM's response and is not directly manipulated by the attacker. We then run the target LLM again on the backtranslated prompt, and we refuse the original prompt if the model refuses the backtranslated prompt. We explain that the proposed defense provides several benefits on its effectiveness and efficiency. We empirically demonstrate that our defense significantly outperforms the baselines, in the cases that are hard for the baselines, and our defense also has little impact on the generation quality for benign input prompts.



## **10. A New Era in LLM Security: Exploring Security Concerns in Real-World LLM-based Systems**

cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18649v1) [paper-pdf](http://arxiv.org/pdf/2402.18649v1)

**Authors**: Fangzhou Wu, Ning Zhang, Somesh Jha, Patrick McDaniel, Chaowei Xiao

**Abstract**: Large Language Model (LLM) systems are inherently compositional, with individual LLM serving as the core foundation with additional layers of objects such as plugins, sandbox, and so on. Along with the great potential, there are also increasing concerns over the security of such probabilistic intelligent systems. However, existing studies on LLM security often focus on individual LLM, but without examining the ecosystem through the lens of LLM systems with other objects (e.g., Frontend, Webtool, Sandbox, and so on). In this paper, we systematically analyze the security of LLM systems, instead of focusing on the individual LLMs. To do so, we build on top of the information flow and formulate the security of LLM systems as constraints on the alignment of the information flow within LLM and between LLM and other objects. Based on this construction and the unique probabilistic nature of LLM, the attack surface of the LLM system can be decomposed into three key components: (1) multi-layer security analysis, (2) analysis of the existence of constraints, and (3) analysis of the robustness of these constraints. To ground this new attack surface, we propose a multi-layer and multi-step approach and apply it to the state-of-art LLM system, OpenAI GPT4. Our investigation exposes several security issues, not just within the LLM model itself but also in its integration with other components. We found that although the OpenAI GPT4 has designed numerous safety constraints to improve its safety features, these safety constraints are still vulnerable to attackers. To further demonstrate the real-world threats of our discovered vulnerabilities, we construct an end-to-end attack where an adversary can illicitly acquire the user's chat history, all without the need to manipulate the user's input or gain direct access to OpenAI GPT4. Our demo is in the link: https://fzwark.github.io/LLM-System-Attack-Demo/



## **11. Multilingual Jailbreak Challenges in Large Language Models**

cs.CL

ICLR 2024

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2310.06474v2) [paper-pdf](http://arxiv.org/pdf/2310.06474v2)

**Authors**: Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, Lidong Bing

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across a wide range of tasks, they pose potential safety concerns, such as the ``jailbreak'' problem, wherein malicious instructions can manipulate LLMs to exhibit undesirable behavior. Although several preventive measures have been developed to mitigate the potential risks associated with LLMs, they have primarily focused on English. In this study, we reveal the presence of multilingual jailbreak challenges within LLMs and consider two potential risky scenarios: unintentional and intentional. The unintentional scenario involves users querying LLMs using non-English prompts and inadvertently bypassing the safety mechanisms, while the intentional scenario concerns malicious users combining malicious instructions with multilingual prompts to deliberately attack LLMs. The experimental results reveal that in the unintentional scenario, the rate of unsafe content increases as the availability of languages decreases. Specifically, low-resource languages exhibit about three times the likelihood of encountering harmful content compared to high-resource languages, with both ChatGPT and GPT-4. In the intentional scenario, multilingual prompts can exacerbate the negative impact of malicious instructions, with astonishingly high rates of unsafe output: 80.92\% for ChatGPT and 40.71\% for GPT-4. To handle such a challenge in the multilingual context, we propose a novel \textsc{Self-Defense} framework that automatically generates multilingual training data for safety fine-tuning. Experimental results show that ChatGPT fine-tuned with such data can achieve a substantial reduction in unsafe content generation. Data is available at \url{https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs}.



## **12. Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction**

cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18104v1) [paper-pdf](http://arxiv.org/pdf/2402.18104v1)

**Authors**: Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, Kai Chen

**Abstract**: In recent years, large language models (LLMs) have demonstrated notable success across various tasks, but the trustworthiness of LLMs is still an open problem. One specific threat is the potential to generate toxic or harmful responses. Attackers can craft adversarial prompts that induce harmful responses from LLMs. In this work, we pioneer a theoretical foundation in LLMs security by identifying bias vulnerabilities within the safety fine-tuning and design a black-box jailbreak method named DRA (Disguise and Reconstruction Attack), which conceals harmful instructions through disguise and prompts the model to reconstruct the original harmful instruction within its completion. We evaluate DRA across various open-source and close-source models, showcasing state-of-the-art jailbreak success rates and attack efficiency. Notably, DRA boasts a 90\% attack success rate on LLM chatbots GPT-4.



## **13. EmMark: Robust Watermarks for IP Protection of Embedded Quantized Large Language Models**

cs.CR

Accept to DAC 2024

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17938v1) [paper-pdf](http://arxiv.org/pdf/2402.17938v1)

**Authors**: Ruisi Zhang, Farinaz Koushanfar

**Abstract**: This paper introduces EmMark,a novel watermarking framework for protecting the intellectual property (IP) of embedded large language models deployed on resource-constrained edge devices. To address the IP theft risks posed by malicious end-users, EmMark enables proprietors to authenticate ownership by querying the watermarked model weights and matching the inserted signatures. EmMark's novelty lies in its strategic watermark weight parameters selection, nsuring robustness and maintaining model quality. Extensive proof-of-concept evaluations of models from OPT and LLaMA-2 families demonstrate EmMark's fidelity, achieving 100% success in watermark extraction with model performance preservation. EmMark also showcased its resilience against watermark removal and forging attacks.



## **14. LLM-Resistant Math Word Problem Generation via Adversarial Attacks**

cs.CL

Code is available at  https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17916v1) [paper-pdf](http://arxiv.org/pdf/2402.17916v1)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis on math problems and investigate the cause of failure to guide future research on LLM's mathematical capability.



## **15. Mitigating Fine-tuning Jailbreak Attack with Backdoor Enhanced Alignment**

cs.CR

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.14968v2) [paper-pdf](http://arxiv.org/pdf/2402.14968v2)

**Authors**: Jiongxiao Wang, Jiazhao Li, Yiquan Li, Xiangyu Qi, Junjie Hu, Yixuan Li, Patrick McDaniel, Muhao Chen, Bo Li, Chaowei Xiao

**Abstract**: Despite the general capabilities of Large Language Models (LLMs) like GPT-4 and Llama-2, these models still request fine-tuning or adaptation with customized data when it comes to meeting the specific business demands and intricacies of tailored use cases. However, this process inevitably introduces new safety threats, particularly against the Fine-tuning based Jailbreak Attack (FJAttack), where incorporating just a few harmful examples into the fine-tuning dataset can significantly compromise the model safety. Though potential defenses have been proposed by incorporating safety examples into the fine-tuning dataset to reduce the safety issues, such approaches require incorporating a substantial amount of safety examples, making it inefficient. To effectively defend against the FJAttack with limited safety examples, we propose a Backdoor Enhanced Safety Alignment method inspired by an analogy with the concept of backdoor attacks. In particular, we construct prefixed safety examples by integrating a secret prompt, acting as a "backdoor trigger", that is prefixed to safety examples. Our comprehensive experiments demonstrate that through the Backdoor Enhanced Safety Alignment with adding as few as 11 prefixed safety examples, the maliciously fine-tuned LLMs will achieve similar safety performance as the original aligned models. Furthermore, we also explore the effectiveness of our method in a more practical setting where the fine-tuning data consists of both FJAttack examples and the fine-tuning task data. Our method shows great efficacy in defending against FJAttack without harming the performance of fine-tuning tasks.



## **16. Semantic Mirror Jailbreak: Genetic Algorithm Based Jailbreak Prompts Against Open-source LLMs**

cs.CL

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.14872v2) [paper-pdf](http://arxiv.org/pdf/2402.14872v2)

**Authors**: Xiaoxia Li, Siyuan Liang, Jiyi Zhang, Han Fang, Aishan Liu, Ee-Chien Chang

**Abstract**: Large Language Models (LLMs), used in creative writing, code generation, and translation, generate text based on input sequences but are vulnerable to jailbreak attacks, where crafted prompts induce harmful outputs. Most jailbreak prompt methods use a combination of jailbreak templates followed by questions to ask to create jailbreak prompts. However, existing jailbreak prompt designs generally suffer from excessive semantic differences, resulting in an inability to resist defenses that use simple semantic metrics as thresholds. Jailbreak prompts are semantically more varied than the original questions used for queries. In this paper, we introduce a Semantic Mirror Jailbreak (SMJ) approach that bypasses LLMs by generating jailbreak prompts that are semantically similar to the original question. We model the search for jailbreak prompts that satisfy both semantic similarity and jailbreak validity as a multi-objective optimization problem and employ a standardized set of genetic algorithms for generating eligible prompts. Compared to the baseline AutoDAN-GA, SMJ achieves attack success rates (ASR) that are at most 35.4% higher without ONION defense and 85.2% higher with ONION defense. SMJ's better performance in all three semantic meaningfulness metrics of Jailbreak Prompt, Similarity, and Outlier, also means that SMJ is resistant to defenses that use those metrics as thresholds.



## **17. HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal**

cs.LG

Website: https://www.harmbench.org

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.04249v2) [paper-pdf](http://arxiv.org/pdf/2402.04249v2)

**Authors**: Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, Dan Hendrycks

**Abstract**: Automated red teaming holds substantial promise for uncovering and mitigating the risks associated with the malicious use of large language models (LLMs), yet the field lacks a standardized evaluation framework to rigorously assess new methods. To address this issue, we introduce HarmBench, a standardized evaluation framework for automated red teaming. We identify several desirable properties previously unaccounted for in red teaming evaluations and systematically design HarmBench to meet these criteria. Using HarmBench, we conduct a large-scale comparison of 18 red teaming methods and 33 target LLMs and defenses, yielding novel insights. We also introduce a highly efficient adversarial training method that greatly enhances LLM robustness across a wide range of attacks, demonstrating how HarmBench enables codevelopment of attacks and defenses. We open source HarmBench at https://github.com/centerforaisafety/HarmBench.



## **18. Pandora's White-Box: Increased Training Data Leakage in Open LLMs**

cs.CR

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.17012v1) [paper-pdf](http://arxiv.org/pdf/2402.17012v1)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we undertake a systematic study of privacy attacks against open source Large Language Models (LLMs), where an adversary has access to either the model weights, gradients, or losses, and tries to exploit them to learn something about the underlying training data. Our headline results are the first membership inference attacks (MIAs) against pre-trained LLMs that are able to simultaneously achieve high TPRs and low FPRs, and a pipeline showing that over $50\%$ (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, customization of the language model, and resources available to the attacker. In the pre-trained setting, we propose three new white-box MIAs: an attack based on the gradient norm, a supervised neural network classifier, and a single step loss ratio attack. All outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and other types of models. In fine-tuning, we find that given access to the loss of the fine-tuned and base models, a fine-tuned loss ratio attack FLoRA is able to achieve near perfect MIA peformance. We then leverage these MIAs to extract fine-tuning data from fine-tuned language models. We find that the pipeline of generating from fine-tuned models prompted with a small snippet of the prefix of each training example, followed by using FLoRa to select the most likely training sample, succeeds the majority of the fine-tuning dataset after only $3$ epochs of fine-tuning. Taken together, these findings show that highly effective MIAs are available in almost all LLM training settings, and highlight that great care must be taken before LLMs are fine-tuned on highly sensitive data and then deployed.



## **19. WIPI: A New Web Threat for LLM-Driven Web Agents**

cs.CR

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.16965v1) [paper-pdf](http://arxiv.org/pdf/2402.16965v1)

**Authors**: Fangzhou Wu, Shutong Wu, Yulong Cao, Chaowei Xiao

**Abstract**: With the fast development of large language models (LLMs), LLM-driven Web Agents (Web Agents for short) have obtained tons of attention due to their superior capability where LLMs serve as the core part of making decisions like the human brain equipped with multiple web tools to actively interact with external deployed websites. As uncountable Web Agents have been released and such LLM systems are experiencing rapid development and drawing closer to widespread deployment in our daily lives, an essential and pressing question arises: "Are these Web Agents secure?". In this paper, we introduce a novel threat, WIPI, that indirectly controls Web Agent to execute malicious instructions embedded in publicly accessible webpages. To launch a successful WIPI works in a black-box environment. This methodology focuses on the form and content of indirect instructions within external webpages, enhancing the efficiency and stealthiness of the attack. To evaluate the effectiveness of the proposed methodology, we conducted extensive experiments using 7 plugin-based ChatGPT Web Agents, 8 Web GPTs, and 3 different open-source Web Agents. The results reveal that our methodology achieves an average attack success rate (ASR) exceeding 90% even in pure black-box scenarios. Moreover, through an ablation study examining various user prefix instructions, we demonstrated that the WIPI exhibits strong robustness, maintaining high performance across diverse prefix instructions.



## **20. CodeChameleon: Personalized Encryption Framework for Jailbreaking Large Language Models**

cs.CL

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.16717v1) [paper-pdf](http://arxiv.org/pdf/2402.16717v1)

**Authors**: Huijie Lv, Xiao Wang, Yuansen Zhang, Caishuang Huang, Shihan Dou, Junjie Ye, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Adversarial misuse, particularly through `jailbreaking' that circumvents a model's safety and ethical protocols, poses a significant challenge for Large Language Models (LLMs). This paper delves into the mechanisms behind such successful attacks, introducing a hypothesis for the safety mechanism of aligned LLMs: intent security recognition followed by response generation. Grounded in this hypothesis, we propose CodeChameleon, a novel jailbreak framework based on personalized encryption tactics. To elude the intent security recognition phase, we reformulate tasks into a code completion format, enabling users to encrypt queries using personalized encryption functions. To guarantee response generation functionality, we embed a decryption function within the instructions, which allows the LLM to decrypt and execute the encrypted queries successfully. We conduct extensive experiments on 7 LLMs, achieving state-of-the-art average Attack Success Rate (ASR). Remarkably, our method achieves an 86.6\% ASR on GPT-4-1106.



## **21. RoCoIns: Enhancing Robustness of Large Language Models through Code-Style Instructions**

cs.CL

Accepted by COLING 2024

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.16431v1) [paper-pdf](http://arxiv.org/pdf/2402.16431v1)

**Authors**: Yuansen Zhang, Xiao Wang, Zhiheng Xi, Han Xia, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Large Language Models (LLMs) have showcased remarkable capabilities in following human instructions. However, recent studies have raised concerns about the robustness of LLMs when prompted with instructions combining textual adversarial samples. In this paper, drawing inspiration from recent works that LLMs are sensitive to the design of the instructions, we utilize instructions in code style, which are more structural and less ambiguous, to replace typically natural language instructions. Through this conversion, we provide LLMs with more precise instructions and strengthen the robustness of LLMs. Moreover, under few-shot scenarios, we propose a novel method to compose in-context demonstrations using both clean and adversarial samples (\textit{adversarial context method}) to further boost the robustness of the LLMs. Experiments on eight robustness datasets show that our method consistently outperforms prompting LLMs with natural language instructions. For example, with gpt-3.5-turbo, our method achieves an improvement of 5.68\% in test set accuracy and a reduction of 5.66 points in Attack Success Rate (ASR).



## **22. Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks**

cs.CL

Accepted in LREC-COLING 2024 - The 2024 Joint International  Conference on Computational Linguistics, Language Resources and Evaluation

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2305.14965v2) [paper-pdf](http://arxiv.org/pdf/2305.14965v2)

**Authors**: Abhinav Rao, Sachin Vashistha, Atharva Naik, Somak Aditya, Monojit Choudhury

**Abstract**: Recent explorations with commercial Large Language Models (LLMs) have shown that non-expert users can jailbreak LLMs by simply manipulating their prompts; resulting in degenerate output behavior, privacy and security breaches, offensive outputs, and violations of content regulator policies. Limited studies have been conducted to formalize and analyze these attacks and their mitigations. We bridge this gap by proposing a formalism and a taxonomy of known (and possible) jailbreaks. We survey existing jailbreak methods and their effectiveness on open-source and commercial LLMs (such as GPT-based models, OPT, BLOOM, and FLAN-T5-XXL). We further discuss the challenges of jailbreak detection in terms of their effectiveness against known attacks. For our analysis, we collect a dataset of 3700 jailbreak prompts across 4 tasks. We will make the dataset public along with the model outputs.



## **23. Immunization against harmful fine-tuning attacks**

cs.CL

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.16382v1) [paper-pdf](http://arxiv.org/pdf/2402.16382v1)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, Jan Batzner, Hassan Sajjad, Frank Rudzicz

**Abstract**: Approaches to aligning large language models (LLMs) with human values has focused on correcting misalignment that emerges from pretraining. However, this focus overlooks another source of misalignment: bad actors might purposely fine-tune LLMs to achieve harmful goals. In this paper, we present an emerging threat model that has arisen from alignment circumvention and fine-tuning attacks. However, lacking in previous works is a clear presentation of the conditions for effective defence. We propose a set of conditions for effective defence against harmful fine-tuning in LLMs called "Immunization conditions," which help us understand how we would construct and measure future defences. Using this formal framework for defence, we offer a synthesis of different research directions that might be persued to prevent harmful fine-tuning attacks and provide a demonstration of how to use these conditions experimentally showing early results of using an adversarial loss to immunize LLama2-7b-chat.



## **24. Privacy-Preserved Neural Graph Databases**

cs.DB

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2312.15591v4) [paper-pdf](http://arxiv.org/pdf/2312.15591v4)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Zihao Wang, Yangqiu Song

**Abstract**: In the era of large language models (LLMs), efficient and accurate data retrieval has become increasingly crucial for the use of domain-specific or private data in the retrieval augmented generation (RAG). Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (GDBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data which can be adaptively trained with LLMs. The usage of neural embedding storage and Complex neural logical Query Answering (CQA) provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the domain-specific or private databases. Malicious attackers can infer more sensitive information in the database using well-designed queries such as from the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training stage due to the privacy concerns. In this work, we propose a privacy-preserved neural graph database (P-NGDB) framework to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to enforce the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries.



## **25. LuaTaint: A Static Taint Analysis System for Web Interface Framework Vulnerability of IoT Devices**

cs.CR

**SubmitDate**: 2024-02-25    [abs](http://arxiv.org/abs/2402.16043v1) [paper-pdf](http://arxiv.org/pdf/2402.16043v1)

**Authors**: Jiahui Xiang, Wenhai Wang, Tong Ye, Peiyu Liu

**Abstract**: IoT devices are currently facing continuous malicious attacks due to their widespread use. Among these IoT devices, web vulnerabilities are also widely exploited because of their inherent characteristics, such as improper permission controls and insecure interfaces. Recently, the embedded system web interface framework has become highly diverse, and specific vulnerabilities can arise if developers forget to detect user input parameters or if the detection process is not strict enough. Therefore, discovering vulnerabilities in the web interfaces of IoT devices accurately and comprehensively through an automated method is a major challenge. This paper aims to work out the challenge. We have developed an automated vulnerability detection system called LuaTaint for the typical web interface framework, LuCI. The system employs static taint analysis to address web security issues on mobile terminal platforms to ensure detection coverage. It integrates rules pertaining to page handler control logic within the taint detection process to improve its extensibility. We also implemented a post-processing step with the assistance of large language models to enhance accuracy and reduce the need for manual analysis. We have created a prototype of LuaTaint and tested it on 92 IoT firmwares from 8 well-known vendors. LuaTaint has discovered 68 unknown vulnerabilities.



## **26. From Noise to Clarity: Unraveling the Adversarial Suffix of Large Language Model Attacks via Translation of Text Embeddings**

cs.CL

**SubmitDate**: 2024-02-25    [abs](http://arxiv.org/abs/2402.16006v1) [paper-pdf](http://arxiv.org/pdf/2402.16006v1)

**Authors**: Hao Wang, Hao Li, Minlie Huang, Lei Sha

**Abstract**: The safety defense methods of Large language models(LLMs) stays limited because the dangerous prompts are manually curated to just few known attack types, which fails to keep pace with emerging varieties. Recent studies found that attaching suffixes to harmful instructions can hack the defense of LLMs and lead to dangerous outputs. This method, while effective, leaves a gap in understanding the underlying mechanics of such adversarial suffix due to the non-readability and it can be relatively easily seen through by common defense methods such as perplexity filters.To cope with this challenge, in this paper, we propose an Adversarial Suffixes Embedding Translation Framework(ASETF) that are able to translate the unreadable adversarial suffixes into coherent, readable text, which makes it easier to understand and analyze the reasons behind harmful content generation by large language models. We conducted experiments on LLMs such as LLaMa2, Vicuna and using the Advbench dataset's harmful instructions. The results indicate that our method achieves a much better attack success rate to existing techniques, while significantly enhancing the textual fluency of the prompts. In addition, our approach can be generalized into a broader method for generating transferable adversarial suffixes that can successfully attack multiple LLMs, even black-box LLMs, such as ChatGPT and Gemini. As a result, the prompts generated through our method exhibit enriched semantic diversity, which potentially provides more adversarial examples for LLM defense methods.



## **27. Safety of Multimodal Large Language Models on Images and Text**

cs.CV

**SubmitDate**: 2024-02-25    [abs](http://arxiv.org/abs/2402.00357v2) [paper-pdf](http://arxiv.org/pdf/2402.00357v2)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Attracted by the impressive power of Multimodal Large Language Models (MLLMs), the public is increasingly utilizing them to improve the efficiency of daily work. Nonetheless, the vulnerabilities of MLLMs to unsafe instructions bring huge safety risks when these models are deployed in real-world scenarios. In this paper, we systematically survey current efforts on the evaluation, attack, and defense of MLLMs' safety on images and text. We begin with introducing the overview of MLLMs on images and text and understanding of safety, which helps researchers know the detailed scope of our survey. Then, we review the evaluation datasets and metrics for measuring the safety of MLLMs. Next, we comprehensively present attack and defense techniques related to MLLMs' safety. Finally, we analyze several unsolved issues and discuss promising research directions. The latest papers are continually collected at https://github.com/isXinLiu/MLLM-Safety-Collection.



## **28. PRP: Propagating Universal Perturbations to Attack Large Language Model Guard-Rails**

cs.CR

**SubmitDate**: 2024-02-24    [abs](http://arxiv.org/abs/2402.15911v1) [paper-pdf](http://arxiv.org/pdf/2402.15911v1)

**Authors**: Neal Mangaokar, Ashish Hooda, Jihye Choi, Shreyas Chandrashekaran, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstract**: Large language models (LLMs) are typically aligned to be harmless to humans. Unfortunately, recent work has shown that such models are susceptible to automated jailbreak attacks that induce them to generate harmful content. More recent LLMs often incorporate an additional layer of defense, a Guard Model, which is a second LLM that is designed to check and moderate the output response of the primary LLM. Our key contribution is to show a novel attack strategy, PRP, that is successful against several open-source (e.g., Llama 2) and closed-source (e.g., GPT 3.5) implementations of Guard Models. PRP leverages a two step prefix-based attack that operates by (a) constructing a universal adversarial prefix for the Guard Model, and (b) propagating this prefix to the response. We find that this procedure is effective across multiple threat models, including ones in which the adversary has no access to the Guard Model at all. Our work suggests that further advances are required on defenses and Guard Models before they can be considered effective.



## **29. On the Safety Concerns of Deploying LLMs/VLMs in Robotics: Highlighting the Risks and Vulnerabilities**

cs.RO

**SubmitDate**: 2024-02-24    [abs](http://arxiv.org/abs/2402.10340v3) [paper-pdf](http://arxiv.org/pdf/2402.10340v3)

**Authors**: Xiyang Wu, Ruiqi Xian, Tianrui Guan, Jing Liang, Souradip Chakraborty, Fuxiao Liu, Brian Sadler, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works have focused on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation, navigation, etc. However, such integration can introduce significant vulnerabilities, in terms of their susceptibility to adversarial attacks due to the language models, potentially leading to catastrophic consequences. By examining recent works at the interface of LLMs/VLMs and robotics, we show that it is easy to manipulate or misguide the robot's actions, leading to safety hazards. We define and provide examples of several plausible adversarial attacks, and conduct experiments on three prominent robot frameworks integrated with a language model, including KnowNo VIMA, and Instruct2Act, to assess their susceptibility to these attacks. Our empirical findings reveal a striking vulnerability of LLM/VLM-robot integrated systems: simple adversarial attacks can significantly undermine the effectiveness of LLM/VLM-robot integrated systems. Specifically, our data demonstrate an average performance deterioration of 21.2% under prompt attacks and a more alarming 30.2% under perception attacks. These results underscore the critical need for robust countermeasures to ensure the safe and reliable deployment of the advanced LLM/VLM-based robotic systems.



## **30. SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding**

cs.CR

**SubmitDate**: 2024-02-24    [abs](http://arxiv.org/abs/2402.08983v2) [paper-pdf](http://arxiv.org/pdf/2402.08983v2)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, Radha Poovendran

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their token probabilities, while simultaneously attenuating the probabilities of token sequences that are aligned with the objectives of jailbreak attacks. We perform extensive experiments on five LLMs using six state-of-the-art jailbreak attacks and four benchmark datasets. Our results show that SafeDecoding significantly reduces the attack success rate and harmfulness of jailbreak attacks without compromising the helpfulness of responses to benign user queries. SafeDecoding outperforms six defense methods.



## **31. LLMs Can Defend Themselves Against Jailbreaking in a Practical Manner: A Vision Paper**

cs.CR

This is a vision paper on defending against LLM jailbreaks

**SubmitDate**: 2024-02-24    [abs](http://arxiv.org/abs/2402.15727v1) [paper-pdf](http://arxiv.org/pdf/2402.15727v1)

**Authors**: Daoyuan Wu, Shuai Wang, Yang Liu, Ning Liu

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs). A considerable amount of research exists proposing more effective jailbreak attacks, including the recent Greedy Coordinate Gradient (GCG) attack, jailbreak template-based attacks such as using "Do-Anything-Now" (DAN), and multilingual jailbreak. In contrast, the defensive side has been relatively less explored. This paper proposes a lightweight yet practical defense called SELFDEFEND, which can defend against all existing jailbreak attacks with minimal delay for jailbreak prompts and negligible delay for normal user prompts. Our key insight is that regardless of the kind of jailbreak strategies employed, they eventually need to include a harmful prompt (e.g., "how to make a bomb") in the prompt sent to LLMs, and we found that existing LLMs can effectively recognize such harmful prompts that violate their safety policies. Based on this insight, we design a shadow stack that concurrently checks whether a harmful prompt exists in the user prompt and triggers a checkpoint in the normal stack once a token of "No" or a harmful prompt is output. The latter could also generate an explainable LLM response to adversarial prompts. We demonstrate our idea of SELFDEFEND works in various jailbreak scenarios through manual analysis in GPT-3.5/4. We also list three future directions to further enhance SELFDEFEND.



## **32. Foot In The Door: Understanding Large Language Model Jailbreaking via Cognitive Psychology**

cs.CL

**SubmitDate**: 2024-02-24    [abs](http://arxiv.org/abs/2402.15690v1) [paper-pdf](http://arxiv.org/pdf/2402.15690v1)

**Authors**: Zhenhua Wang, Wei Xie, Baosheng Wang, Enze Wang, Zhiwen Gui, Shuoyoucheng Ma, Kai Chen

**Abstract**: Large Language Models (LLMs) have gradually become the gateway for people to acquire new knowledge. However, attackers can break the model's security protection ("jail") to access restricted information, which is called "jailbreaking." Previous studies have shown the weakness of current LLMs when confronted with such jailbreaking attacks. Nevertheless, comprehension of the intrinsic decision-making mechanism within the LLMs upon receipt of jailbreak prompts is noticeably lacking. Our research provides a psychological explanation of the jailbreak prompts. Drawing on cognitive consistency theory, we argue that the key to jailbreak is guiding the LLM to achieve cognitive coordination in an erroneous direction. Further, we propose an automatic black-box jailbreaking method based on the Foot-in-the-Door (FITD) technique. This method progressively induces the model to answer harmful questions via multi-step incremental prompts. We instantiated a prototype system to evaluate the jailbreaking effectiveness on 8 advanced LLMs, yielding an average success rate of 83.9%. This study builds a psychological perspective on the explanatory insights into the intrinsic decision-making logic of LLMs.



## **33. User Inference Attacks on Large Language Models**

cs.CR

v2 contains experiments on additional datasets and differential  privacy

**SubmitDate**: 2024-02-23    [abs](http://arxiv.org/abs/2310.09266v2) [paper-pdf](http://arxiv.org/pdf/2310.09266v2)

**Authors**: Nikhil Kandpal, Krishna Pillutla, Alina Oprea, Peter Kairouz, Christopher A. Choquette-Choo, Zheng Xu

**Abstract**: Fine-tuning is a common and effective method for tailoring large language models (LLMs) to specialized tasks and applications. In this paper, we study the privacy implications of fine-tuning LLMs on user data. To this end, we consider a realistic threat model, called user inference, wherein an attacker infers whether or not a user's data was used for fine-tuning. We design attacks for performing user inference that require only black-box access to the fine-tuned LLM and a few samples from a user which need not be from the fine-tuning dataset. We find that LLMs are susceptible to user inference across a variety of fine-tuning datasets, at times with near perfect attack success rates. Further, we theoretically and empirically investigate the properties that make users vulnerable to user inference, finding that outlier users, users with identifiable shared features between examples, and users that contribute a large fraction of the fine-tuning data are most susceptible to attack. Based on these findings, we identify several methods for mitigating user inference including training with example-level differential privacy, removing within-user duplicate examples, and reducing a user's contribution to the training data. While these techniques provide partial mitigation of user inference, we highlight the need to develop methods to fully protect fine-tuned LLMs against this privacy risk.



## **34. The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG)**

cs.CR

**SubmitDate**: 2024-02-23    [abs](http://arxiv.org/abs/2402.16893v1) [paper-pdf](http://arxiv.org/pdf/2402.16893v1)

**Authors**: Shenglai Zeng, Jiankun Zhang, Pengfei He, Yue Xing, Yiding Liu, Han Xu, Jie Ren, Shuaiqiang Wang, Dawei Yin, Yi Chang, Jiliang Tang

**Abstract**: Retrieval-augmented generation (RAG) is a powerful technique to facilitate language model with proprietary and private data, where data privacy is a pivotal concern. Whereas extensive research has demonstrated the privacy risks of large language models (LLMs), the RAG technique could potentially reshape the inherent behaviors of LLM generation, posing new privacy issues that are currently under-explored. In this work, we conduct extensive empirical studies with novel attack methods, which demonstrate the vulnerability of RAG systems on leaking the private retrieval database. Despite the new risk brought by RAG on the retrieval data, we further reveal that RAG can mitigate the leakage of the LLMs' training data. Overall, we provide new insights in this paper for privacy protection of retrieval-augmented LLMs, which benefit both LLMs and RAG systems builders. Our code is available at https://github.com/phycholosogy/RAG-privacy.



## **35. Analyzing the Inherent Response Tendency of LLMs: Real-World Instructions-Driven Jailbreak**

cs.CL

**SubmitDate**: 2024-02-23    [abs](http://arxiv.org/abs/2312.04127v2) [paper-pdf](http://arxiv.org/pdf/2312.04127v2)

**Authors**: Yanrui Du, Sendong Zhao, Ming Ma, Yuhan Chen, Bing Qin

**Abstract**: Extensive work has been devoted to improving the safety mechanism of Large Language Models (LLMs). However, LLMs still tend to generate harmful responses when faced with malicious instructions, a phenomenon referred to as "Jailbreak Attack". In our research, we introduce a novel automatic jailbreak method RADIAL, which bypasses the security mechanism by amplifying the potential of LLMs to generate affirmation responses. The jailbreak idea of our method is "Inherent Response Tendency Analysis" which identifies real-world instructions that can inherently induce LLMs to generate affirmation responses and the corresponding jailbreak strategy is "Real-World Instructions-Driven Jailbreak" which involves strategically splicing real-world instructions identified through the above analysis around the malicious instruction. Our method achieves excellent attack performance on English malicious instructions with five open-source advanced LLMs while maintaining robust attack performance in executing cross-language attacks against Chinese malicious instructions. We conduct experiments to verify the effectiveness of our jailbreak idea and the rationality of our jailbreak strategy design. Notably, our method designed a semantically coherent attack prompt, highlighting the potential risks of LLMs. Our study provides detailed insights into jailbreak attacks, establishing a foundation for the development of safer LLMs.



## **36. A First Look at GPT Apps: Landscape and Vulnerability**

cs.CR

**SubmitDate**: 2024-02-23    [abs](http://arxiv.org/abs/2402.15105v1) [paper-pdf](http://arxiv.org/pdf/2402.15105v1)

**Authors**: Zejun Zhang, Li Zhang, Xin Yuan, Anlan Zhang, Mengwei Xu, Feng Qian

**Abstract**: With the advancement of Large Language Models (LLMs), increasingly sophisticated and powerful GPTs are entering the market. Despite their popularity, the LLM ecosystem still remains unexplored. Additionally, LLMs' susceptibility to attacks raises concerns over safety and plagiarism. Thus, in this work, we conduct a pioneering exploration of GPT stores, aiming to study vulnerabilities and plagiarism within GPT applications. To begin with, we conduct, to our knowledge, the first large-scale monitoring and analysis of two stores, an unofficial GPTStore.AI, and an official OpenAI GPT Store. Then, we propose a TriLevel GPT Reversing (T-GR) strategy for extracting GPT internals. To complete these two tasks efficiently, we develop two automated tools: one for web scraping and another designed for programmatically interacting with GPTs. Our findings reveal a significant enthusiasm among users and developers for GPT interaction and creation, as evidenced by the rapid increase in GPTs and their creators. However, we also uncover a widespread failure to protect GPT internals, with nearly 90% of system prompts easily accessible, leading to considerable plagiarism and duplication among GPTs.



## **37. ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs**

cs.CL

**SubmitDate**: 2024-02-22    [abs](http://arxiv.org/abs/2402.11753v2) [paper-pdf](http://arxiv.org/pdf/2402.11753v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Zhen Xiang, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran

**Abstract**: Safety is critical to the usage of large language models (LLMs). Multiple techniques such as data filtering and supervised fine-tuning have been developed to strengthen LLM safety. However, currently known techniques presume that corpora used for safety alignment of LLMs are solely interpreted by semantics. This assumption, however, does not hold in real-world applications, which leads to severe vulnerabilities in LLMs. For example, users of forums often use ASCII art, a form of text-based art, to convey image information. In this paper, we propose a novel ASCII art-based jailbreak attack and introduce a comprehensive benchmark Vision-in-Text Challenge (ViTC) to evaluate the capabilities of LLMs in recognizing prompts that cannot be solely interpreted by semantics. We show that five SOTA LLMs (GPT-3.5, GPT-4, Gemini, Claude, and Llama2) struggle to recognize prompts provided in the form of ASCII art. Based on this observation, we develop the jailbreak attack ArtPrompt, which leverages the poor performance of LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs. ArtPrompt only requires black-box access to the victim LLMs, making it a practical attack. We evaluate ArtPrompt on five SOTA LLMs, and show that ArtPrompt can effectively and efficiently induce undesired behaviors from all five LLMs.



## **38. Coercing LLMs to do and reveal (almost) anything**

cs.LG

32 pages. Implementation available at  https://github.com/JonasGeiping/carving

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.14020v1) [paper-pdf](http://arxiv.org/pdf/2402.14020v1)

**Authors**: Jonas Geiping, Alex Stein, Manli Shu, Khalid Saifullah, Yuxin Wen, Tom Goldstein

**Abstract**: It has recently been shown that adversarial attacks on large language models (LLMs) can "jailbreak" the model into making harmful statements. In this work, we argue that the spectrum of adversarial attacks on LLMs is much larger than merely jailbreaking. We provide a broad overview of possible attack surfaces and attack goals. Based on a series of concrete examples, we discuss, categorize and systematize attacks that coerce varied unintended behaviors, such as misdirection, model control, denial-of-service, or data extraction.   We analyze these attacks in controlled experiments, and find that many of them stem from the practice of pre-training LLMs with coding capabilities, as well as the continued existence of strange "glitch" tokens in common LLM vocabularies that should be removed for security reasons.



## **39. Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment**

cs.CL

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.14016v1) [paper-pdf](http://arxiv.org/pdf/2402.14016v1)

**Authors**: Vyas Raina, Adian Liusie, Mark Gales

**Abstract**: Large Language Models (LLMs) are powerful zero-shot assessors and are increasingly used in real-world situations such as for written exams or benchmarking systems. Despite this, no existing work has analyzed the vulnerability of judge-LLMs against adversaries attempting to manipulate outputs. This work presents the first study on the adversarial robustness of assessment LLMs, where we search for short universal phrases that when appended to texts can deceive LLMs to provide high assessment scores. Experiments on SummEval and TopicalChat demonstrate that both LLM-scoring and pairwise LLM-comparative assessment are vulnerable to simple concatenation attacks, where in particular LLM-scoring is very susceptible and can yield maximum assessment scores irrespective of the input text quality. Interestingly, such attacks are transferable and phrases learned on smaller open-source LLMs can be applied to larger closed-source models, such as GPT3.5. This highlights the pervasive nature of the adversarial vulnerabilities across different judge-LLM sizes, families and methods. Our findings raise significant concerns on the reliability of LLMs-as-a-judge methods, and underscore the importance of addressing vulnerabilities in LLM assessment methods before deployment in high-stakes real-world scenarios.



## **40. Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models**

cs.CL

Under review

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.14007v1) [paper-pdf](http://arxiv.org/pdf/2402.14007v1)

**Authors**: Zhiwei He, Binglin Zhou, Hongkun Hao, Aiwei Liu, Xing Wang, Zhaopeng Tu, Zhuosheng Zhang, Rui Wang

**Abstract**: Text watermarking technology aims to tag and identify content produced by large language models (LLMs) to prevent misuse. In this study, we introduce the concept of ''cross-lingual consistency'' in text watermarking, which assesses the ability of text watermarks to maintain their effectiveness after being translated into other languages. Preliminary empirical results from two LLMs and three watermarking methods reveal that current text watermarking technologies lack consistency when texts are translated into various languages. Based on this observation, we propose a Cross-lingual Watermark Removal Attack (CWRA) to bypass watermarking by first obtaining a response from an LLM in a pivot language, which is then translated into the target language. CWRA can effectively remove watermarks by reducing the Area Under the Curve (AUC) from 0.95 to 0.67 without performance loss. Furthermore, we analyze two key factors that contribute to the cross-lingual consistency in text watermarking and propose a defense method that increases the AUC from 0.67 to 0.88 under CWRA.



## **41. Tree of Attacks: Jailbreaking Black-Box LLMs Automatically**

cs.LG

An implementation of the presented method is available at  https://github.com/RICommunity/TAP

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2312.02119v2) [paper-pdf](http://arxiv.org/pdf/2312.02119v2)

**Authors**: Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi

**Abstract**: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thought reasoning until one of the generated prompts jailbreaks the target. Crucially, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning reduces the total number of queries sent to the target. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo) for more than 80% of the prompts using only a small number of queries. Interestingly, TAP is also capable of jailbreaking LLMs protected by state-of-the-art guardrails, e.g., LlamaGuard. This significantly improves upon the previous state-of-the-art black-box method for generating jailbreaks.



## **42. Large Language Models are Vulnerable to Bait-and-Switch Attacks for Generating Harmful Content**

cs.CL

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.13926v1) [paper-pdf](http://arxiv.org/pdf/2402.13926v1)

**Authors**: Federico Bianchi, James Zou

**Abstract**: The risks derived from large language models (LLMs) generating deceptive and damaging content have been the subject of considerable research, but even safe generations can lead to problematic downstream impacts. In our study, we shift the focus to how even safe text coming from LLMs can be easily turned into potentially dangerous content through Bait-and-Switch attacks. In such attacks, the user first prompts LLMs with safe questions and then employs a simple find-and-replace post-hoc technique to manipulate the outputs into harmful narratives. The alarming efficacy of this approach in generating toxic content highlights a significant challenge in developing reliable safety guardrails for LLMs. In particular, we stress that focusing on the safety of the verbatim LLM outputs is insufficient and that we also need to consider post-hoc transformations.



## **43. Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!**

cs.CL

Project web page: https://zhziszz.github.io/emulated-disalignment

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.12343v2) [paper-pdf](http://arxiv.org/pdf/2402.12343v2)

**Authors**: Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao

**Abstract**: Large language models (LLMs) need to undergo safety alignment to ensure safe conversations with humans. However, in this work, we introduce an inference-time attack framework, demonstrating that safety alignment can also unintentionally facilitate harmful outcomes under adversarial manipulation. This framework, named Emulated Disalignment (ED), adversely combines a pair of open-source pre-trained and safety-aligned language models in the output space to produce a harmful language model without additional training. Our experiments with ED across three datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show that ED doubles the harmfulness of pre-trained models and outperforms strong baselines, achieving the highest harmful rate in 43 out of 48 evaluation subsets by a large margin. Crucially, our findings highlight the importance of reevaluating the practice of open-sourcing language models even after safety alignment.



## **44. An Explainable Transformer-based Model for Phishing Email Detection: A Large Language Model Approach**

cs.LG

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.13871v1) [paper-pdf](http://arxiv.org/pdf/2402.13871v1)

**Authors**: Mohammad Amaz Uddin, Iqbal H. Sarker

**Abstract**: Phishing email is a serious cyber threat that tries to deceive users by sending false emails with the intention of stealing confidential information or causing financial harm. Attackers, often posing as trustworthy entities, exploit technological advancements and sophistication to make detection and prevention of phishing more challenging. Despite extensive academic research, phishing detection remains an ongoing and formidable challenge in the cybersecurity landscape. Large Language Models (LLMs) and Masked Language Models (MLMs) possess immense potential to offer innovative solutions to address long-standing challenges. In this research paper, we present an optimized, fine-tuned transformer-based DistilBERT model designed for the detection of phishing emails. In the detection process, we work with a phishing email dataset and utilize the preprocessing techniques to clean and solve the imbalance class issues. Through our experiments, we found that our model effectively achieves high accuracy, demonstrating its capability to perform well. Finally, we demonstrate our fine-tuned model using Explainable-AI (XAI) techniques such as Local Interpretable Model-Agnostic Explanations (LIME) and Transformer Interpret to explain how our model makes predictions in the context of text classification for phishing emails.



## **45. Intention Analysis Makes LLMs A Good Jailbreak Defender**

cs.CL

17 pages, 12 figures

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2401.06561v2) [paper-pdf](http://arxiv.org/pdf/2401.06561v2)

**Authors**: Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao

**Abstract**: Aligning large language models (LLMs) with human values, particularly in the face of stealthy and complex jailbreak attacks, presents a formidable challenge. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis ($\mathbb{IA}$). The principle behind this is to trigger LLMs' inherent self-correct and improve ability through a two-stage process: 1) essential intention analysis, and 2) policy-aligned response. Notably, $\mathbb{IA}$ is an inference-only method, thus could enhance the safety of LLMs without compromising their helpfulness. Extensive experiments on SAP200 and DAN benchmarks across Vicuna, ChatGLM, MPT, DeepSeek, and GPT-3.5 show that $\mathbb{IA}$ could consistently and significantly reduce the harmfulness in responses (averagely -46.5\% attack success rate) and maintain the general helpfulness. Encouragingly, with the help of our $\mathbb{IA}$, Vicuna-7b even outperforms GPT-3.5 in terms of attack success rate. Further analyses present some insights into how our method works. To facilitate reproducibility, we release our code and scripts at: https://github.com/alphadl/SafeLLM_with_IntentionAnalysis.



## **46. Round Trip Translation Defence against Large Language Model Jailbreaking Attacks**

cs.CL

6 pages, 6 figures

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.13517v1) [paper-pdf](http://arxiv.org/pdf/2402.13517v1)

**Authors**: Canaan Yung, Hadi Mohaghegh Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Large language models (LLMs) are susceptible to social-engineered attacks that are human-interpretable but require a high level of comprehension for LLMs to counteract. Existing defensive measures can only mitigate less than half of these attacks at most. To address this issue, we propose the Round Trip Translation (RTT) method, the first algorithm specifically designed to defend against social-engineered attacks on LLMs. RTT paraphrases the adversarial prompt and generalizes the idea conveyed, making it easier for LLMs to detect induced harmful behavior. This method is versatile, lightweight, and transferrable to different LLMs. Our defense successfully mitigated over 70% of Prompt Automatic Iterative Refinement (PAIR) attacks, which is currently the most effective defense to the best of our knowledge. We are also the first to attempt mitigating the MathsAttack and reduced its attack success rate by almost 40%. Our code is publicly available at https://github.com/Cancanxxx/Round_Trip_Translation_Defence



## **47. Learning to Poison Large Language Models During Instruction Tuning**

cs.LG

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.13459v1) [paper-pdf](http://arxiv.org/pdf/2402.13459v1)

**Authors**: Yao Qiang, Xiangyu Zhou, Saleh Zare Zade, Mohammad Amin Roshani, Douglas Zytko, Dongxiao Zhu

**Abstract**: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where adversaries insert backdoor triggers into training data to manipulate outputs for malicious purposes. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the instruction tuning process. We propose a novel gradient-guided backdoor trigger learning approach to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various LLMs and tasks, our strategy demonstrates a high success rate in compromising model outputs; poisoning only 1\% of 4,000 instruction tuning samples leads to a Performance Drop Rate (PDR) of around 80\%. Our work highlights the need for stronger defenses against data poisoning attack, offering insights into safeguarding LLMs against these more sophisticated attacks. The source code can be found on this GitHub repository: https://github.com/RookieZxy/GBTL/blob/main/README.md.



## **48. LLM Jailbreak Attack versus Defense Techniques -- A Comprehensive Study**

cs.CR

16 pages, 6 figures

**SubmitDate**: 2024-02-21    [abs](http://arxiv.org/abs/2402.13457v1) [paper-pdf](http://arxiv.org/pdf/2402.13457v1)

**Authors**: Zihao Xu, Yi Liu, Gelei Deng, Yuekang Li, Stjepan Picek

**Abstract**: Large Language Models (LLMS) have increasingly become central to generating content with potential societal impacts. Notably, these models have demonstrated capabilities for generating content that could be deemed harmful. To mitigate these risks, researchers have adopted safety training techniques to align model outputs with societal values to curb the generation of malicious content. However, the phenomenon of "jailbreaking", where carefully crafted prompts elicit harmful responses from models, persists as a significant challenge. This research conducts a comprehensive analysis of existing studies on jailbreaking LLMs and their defense techniques. We meticulously investigate nine attack techniques and seven defense techniques applied across three distinct language models: Vicuna, LLama, and GPT-3.5 Turbo. We aim to evaluate the effectiveness of these attack and defense techniques. Our findings reveal that existing white-box attacks underperform compared to universal techniques and that including special tokens in the input significantly affects the likelihood of successful attacks. This research highlights the need to concentrate on the security facets of LLMs. Additionally, we contribute to the field by releasing our datasets and testing framework, aiming to foster further research into LLM security. We believe these contributions will facilitate the exploration of security measures within this domain.



## **49. Defending Jailbreak Prompts via In-Context Adversarial Game**

cs.LG

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.13148v1) [paper-pdf](http://arxiv.org/pdf/2402.13148v1)

**Authors**: Yujun Zhou, Yufei Han, Haomin Zhuang, Taicheng Guo, Kehan Guo, Zhenwen Liang, Hongyan Bao, Xiangliang Zhang

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Moreover, ICAG demonstrates remarkable transferability to other LLMs, indicating its potential as a versatile defense mechanism.



## **50. Humans or LLMs as the Judge? A Study on Judgement Biases**

cs.CL

19 pages

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.10669v2) [paper-pdf](http://arxiv.org/pdf/2402.10669v2)

**Authors**: Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, Benyou Wang

**Abstract**: Adopting human and large language models (LLM) as judges (\textit{a.k.a} human- and LLM-as-a-judge) for evaluating the performance of existing LLMs has recently gained attention. Nonetheless, this approach concurrently introduces potential biases from human and LLM judges, questioning the reliability of the evaluation results. In this paper, we propose a novel framework for investigating 5 types of biases for LLM and human judges. We curate a dataset with 142 samples referring to the revised Bloom's Taxonomy and conduct thousands of human and LLM evaluations. Results show that human and LLM judges are vulnerable to perturbations to various degrees, and that even the most cutting-edge judges possess considerable biases. We further exploit their weakness and conduct attacks on LLM judges. We hope that our work can notify the community of the vulnerability of human- and LLM-as-a-judge against perturbations, as well as the urgency of developing robust evaluation systems.



