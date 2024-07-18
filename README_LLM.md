# Latest Large Language Model Attack Papers
**update at 2024-07-18 16:49:17**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Counterfactual Explainable Incremental Prompt Attack Analysis on Large Language Models**

cs.CR

23 pages, 6 figures

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.09292v2) [paper-pdf](http://arxiv.org/pdf/2407.09292v2)

**Authors**: Dong Shu, Mingyu Jin, Tianle Chen, Chong Zhang, Yongfeng Zhang

**Abstract**: This study sheds light on the imperative need to bolster safety and privacy measures in large language models (LLMs), such as GPT-4 and LLaMA-2, by identifying and mitigating their vulnerabilities through explainable analysis of prompt attacks. We propose Counterfactual Explainable Incremental Prompt Attack (CEIPA), a novel technique where we guide prompts in a specific manner to quantitatively measure attack effectiveness and explore the embedded defense mechanisms in these models. Our approach is distinctive for its capacity to elucidate the reasons behind the generation of harmful responses by LLMs through an incremental counterfactual methodology. By organizing the prompt modification process into four incremental levels: (word, sentence, character, and a combination of character and word) we facilitate a thorough examination of the susceptibilities inherent to LLMs. The findings from our study not only provide counterfactual explanation insight but also demonstrate that our framework significantly enhances the effectiveness of attack prompts.



## **2. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

cs.CL

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2405.06134v2) [paper-pdf](http://arxiv.org/pdf/2405.06134v2)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<|endoftext|>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<|endoftext|>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.



## **3. Security Matrix for Multimodal Agents on Mobile Devices: A Systematic and Proof of Concept Study**

cs.CR

Preprint. Work in progress

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.09295v2) [paper-pdf](http://arxiv.org/pdf/2407.09295v2)

**Authors**: Yulong Yang, Xinshan Yang, Shuaidong Li, Chenhao Lin, Zhengyu Zhao, Chao Shen, Tianwei Zhang

**Abstract**: The rapid progress in the reasoning capability of the Multi-modal Large Language Models (MLLMs) has triggered the development of autonomous agent systems on mobile devices. MLLM-based mobile agent systems consist of perception, reasoning, memory, and multi-agent collaboration modules, enabling automatic analysis of user instructions and the design of task pipelines with only natural language and device screenshots as inputs. Despite the increased human-machine interaction efficiency, the security risks of MLLM-based mobile agent systems have not been systematically studied. Existing security benchmarks for agents mainly focus on Web scenarios, and the attack techniques against MLLMs are also limited in the mobile agent scenario. To close these gaps, this paper proposes a mobile agent security matrix covering 3 functional modules of the agent systems. Based on the security matrix, this paper proposes 4 realistic attack paths and verifies these attack paths through 8 attack methods. By analyzing the attack results, this paper reveals that MLLM-based mobile agent systems are not only vulnerable to multiple traditional attacks, but also raise new security concerns previously unconsidered. This paper highlights the need for security awareness in the design of MLLM-based systems and paves the way for future research on attacks and defense methods.



## **4. DistillSeq: A Framework for Safety Alignment Testing in Large Language Models using Knowledge Distillation**

cs.SE

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.10106v2) [paper-pdf](http://arxiv.org/pdf/2407.10106v2)

**Authors**: Mingke Yang, Yuqi Chen, Yi Liu, Ling Shi

**Abstract**: Large Language Models (LLMs) have showcased their remarkable capabilities in diverse domains, encompassing natural language understanding, translation, and even code generation. The potential for LLMs to generate harmful content is a significant concern. This risk necessitates rigorous testing and comprehensive evaluation of LLMs to ensure safe and responsible use. However, extensive testing of LLMs requires substantial computational resources, making it an expensive endeavor. Therefore, exploring cost-saving strategies during the testing phase is crucial to balance the need for thorough evaluation with the constraints of resource availability. To address this, our approach begins by transferring the moderation knowledge from an LLM to a small model. Subsequently, we deploy two distinct strategies for generating malicious queries: one based on a syntax tree approach, and the other leveraging an LLM-based method. Finally, our approach incorporates a sequential filter-test process designed to identify test cases that are prone to eliciting toxic responses. Our research evaluated the efficacy of DistillSeq across four LLMs: GPT-3.5, GPT-4.0, Vicuna-13B, and Llama-13B. In the absence of DistillSeq, the observed attack success rates on these LLMs stood at 31.5% for GPT-3.5, 21.4% for GPT-4.0, 28.3% for Vicuna-13B, and 30.9% for Llama-13B. However, upon the application of DistillSeq, these success rates notably increased to 58.5%, 50.7%, 52.5%, and 54.4%, respectively. This translated to an average escalation in attack success rate by a factor of 93.0% when compared to scenarios without the use of DistillSeq. Such findings highlight the significant enhancement DistillSeq offers in terms of reducing the time and resource investment required for effectively testing LLMs.



## **5. The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?**

cs.CV

ECCV 2024. Project page: https://github.com/Qinyu-Allen-Zhao/LVLM-LP

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2403.09037v2) [paper-pdf](http://arxiv.org/pdf/2403.09037v2)

**Authors**: Qinyu Zhao, Ming Xu, Kartik Gupta, Akshay Asthana, Liang Zheng, Stephen Gould

**Abstract**: Large vision-language models (LVLMs), designed to interpret and respond to human instructions, occasionally generate hallucinated or harmful content due to inappropriate instructions. This study uses linear probing to shed light on the hidden knowledge at the output layers of LVLMs. We demonstrate that the logit distributions of the first tokens contain sufficient information to determine whether to respond to the instructions, including recognizing unanswerable visual questions, defending against jailbreaking attacks, and identifying deceptive questions. Such hidden knowledge is gradually lost in logits of subsequent tokens during response generation. Then, we illustrate a simple decoding strategy at the generation of the first token, effectively improving the generated content. In experiments, we find a few interesting insights: First, the CLIP model already contains a strong signal for solving these tasks, which indicates potential bias in the existing datasets. Second, we observe performance improvement by utilizing the first logit distributions on three additional tasks, including indicating uncertainty in math solving, mitigating hallucination, and image classification. Last, with the same training data, simply finetuning LVLMs improves models' performance but is still inferior to linear probing on these tasks.



## **6. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

cs.CV

16 pages, 14 figures

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2404.19287v2) [paper-pdf](http://arxiv.org/pdf/2404.19287v2)

**Authors**: Wanqi Zhou, Shuanghao Bai, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP have shown impressive generalization performance across various downstream tasks, yet they remain vulnerable to adversarial attacks. While prior research has primarily concentrated on improving the adversarial robustness of image encoders to guard against attacks on images, the exploration of text-based and multimodal attacks has largely been overlooked. In this work, we initiate the first known and comprehensive effort to study adapting vision-language models for adversarial robustness under the multimodal attack. Firstly, we introduce a multimodal attack strategy and investigate the impact of different attacks. We then propose a multimodal contrastive adversarial training loss, aligning the clean and adversarial text embeddings with the adversarial and clean visual features, to enhance the adversarial robustness of both image and text encoders of CLIP. Extensive experiments on 15 datasets across two tasks demonstrate that our method significantly improves the adversarial robustness of CLIP. Interestingly, we find that the model fine-tuned against multimodal adversarial attacks exhibits greater robustness than its counterpart fine-tuned solely against image-based attacks, even in the context of image attacks, which may open up new possibilities for enhancing the security of VLMs.



## **7. Turning Generative Models Degenerate: The Power of Data Poisoning Attacks**

cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.12281v1) [paper-pdf](http://arxiv.org/pdf/2407.12281v1)

**Authors**: Shuli Jiang, Swanand Ravindra Kadhe, Yi Zhou, Farhan Ahmed, Ling Cai, Nathalie Baracaldo

**Abstract**: The increasing use of large language models (LLMs) trained by third parties raises significant security concerns. In particular, malicious actors can introduce backdoors through poisoning attacks to generate undesirable outputs. While such attacks have been extensively studied in image domains and classification tasks, they remain underexplored for natural language generation (NLG) tasks. To address this gap, we conduct an investigation of various poisoning techniques targeting the LLM's fine-tuning phase via prefix-tuning, a Parameter Efficient Fine-Tuning (PEFT) method. We assess their effectiveness across two generative tasks: text summarization and text completion; and we also introduce new metrics to quantify the success and stealthiness of such NLG poisoning attacks. Through our experiments, we find that the prefix-tuning hyperparameters and trigger designs are the most crucial factors to influence attack success and stealthiness. Moreover, we demonstrate that existing popular defenses are ineffective against our poisoning attacks. Our study presents the first systematic approach to understanding poisoning attacks targeting NLG tasks during fine-tuning via PEFT across a wide range of triggers and attack settings. We hope our findings will aid the AI security community in developing effective defenses against such threats.



## **8. Uncertainty is Fragile: Manipulating Uncertainty in Large Language Models**

cs.CL

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.11282v2) [paper-pdf](http://arxiv.org/pdf/2407.11282v2)

**Authors**: Qingcheng Zeng, Mingyu Jin, Qinkai Yu, Zhenting Wang, Wenyue Hua, Zihao Zhou, Guangyan Sun, Yanda Meng, Shiqing Ma, Qifan Wang, Felix Juefei-Xu, Kaize Ding, Fan Yang, Ruixiang Tang, Yongfeng Zhang

**Abstract**: Large Language Models (LLMs) are employed across various high-stakes domains, where the reliability of their outputs is crucial. One commonly used method to assess the reliability of LLMs' responses is uncertainty estimation, which gauges the likelihood of their answers being correct. While many studies focus on improving the accuracy of uncertainty estimations for LLMs, our research investigates the fragility of uncertainty estimation and explores potential attacks. We demonstrate that an attacker can embed a backdoor in LLMs, which, when activated by a specific trigger in the input, manipulates the model's uncertainty without affecting the final output. Specifically, the proposed backdoor attack method can alter an LLM's output probability distribution, causing the probability distribution to converge towards an attacker-predefined distribution while ensuring that the top-1 prediction remains unchanged. Our experimental results demonstrate that this attack effectively undermines the model's self-evaluation reliability in multiple-choice questions. For instance, we achieved a 100 attack success rate (ASR) across three different triggering strategies in four models. Further, we investigate whether this manipulation generalizes across different prompts and domains. This work highlights a significant threat to the reliability of LLMs and underscores the need for future defenses against such attacks. The code is available at https://github.com/qcznlp/uncertainty_attack.



## **9. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

JailbreakBench v1.0: more attack artifacts, more test-time defenses,  a more accurate jailbreak judge (Llama-3-70B with a custom prompt), a larger  dataset of human preferences for selecting a jailbreak judge (300 examples),  an over-refusal evaluation dataset (100 benign/borderline behaviors), a  semantic refusal judge based on Llama-3-8B

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2404.01318v4) [paper-pdf](http://arxiv.org/pdf/2404.01318v4)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.



## **10. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

cs.MM

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2405.19802v3) [paper-pdf](http://arxiv.org/pdf/2405.19802v3)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.



## **11. Robust Utility-Preserving Text Anonymization Based on Large Language Models**

cs.CL

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11770v1) [paper-pdf](http://arxiv.org/pdf/2407.11770v1)

**Authors**: Tianyu Yang, Xiaodan Zhu, Iryna Gurevych

**Abstract**: Text anonymization is crucial for sharing sensitive data while maintaining privacy. Existing techniques face the emerging challenges of re-identification attack ability of Large Language Models (LLMs), which have shown advanced capability in memorizing detailed information and patterns as well as connecting disparate pieces of information. In defending against LLM-based re-identification attacks, anonymization could jeopardize the utility of the resulting anonymized data in downstream tasks -- the trade-off between privacy and data utility requires deeper understanding within the context of LLMs. This paper proposes a framework composed of three LLM-based components -- a privacy evaluator, a utility evaluator, and an optimization component, which work collaboratively to perform anonymization. To provide a practical model for large-scale and real-time environments, we distill the anonymization capabilities into a lightweight model using Direct Preference Optimization (DPO). Extensive experiments demonstrate that the proposed models outperform baseline models, showing robustness in reducing the risk of re-identification while preserving greater data utility in downstream tasks. Our code and dataset are available at https://github.com/UKPLab/arxiv2024-rupta.



## **12. Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness**

cs.LG

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.12068v1) [paper-pdf](http://arxiv.org/pdf/2407.12068v1)

**Authors**: Kai Guo, Zewen Liu, Zhikai Chen, Hongzhi Wen, Wei Jin, Jiliang Tang, Yi Chang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.



## **13. PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

cs.CL

Technical report; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2306.04528v5) [paper-pdf](http://arxiv.org/pdf/2306.04528v5)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptRobust, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. The adversarial prompts, crafted to mimic plausible user errors like typos or synonyms, aim to evaluate how slight deviations can affect LLM outcomes while maintaining semantic integrity. These prompts are then employed in diverse tasks including sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,788 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets. Our findings demonstrate that contemporary LLMs are not robust to adversarial prompts. Furthermore, we present a comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users.



## **14. Static Detection of Filesystem Vulnerabilities in Android Systems**

cs.CR

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.11279v1) [paper-pdf](http://arxiv.org/pdf/2407.11279v1)

**Authors**: Yu-Tsung Lee, Hayawardh Vijayakumar, Zhiyun Qian, Trent Jaeger

**Abstract**: Filesystem vulnerabilities persist as a significant threat to Android systems, despite various proposed defenses and testing techniques. The complexity of program behaviors and access control mechanisms in Android systems makes it challenging to effectively identify these vulnerabilities. In this paper, we present PathSentinel, which overcomes the limitations of previous techniques by combining static program analysis and access control policy analysis to detect three types of filesystem vulnerabilities: path traversals, hijacking vulnerabilities, and luring vulnerabilities. By unifying program and access control policy analysis, PathSentinel identifies attack surfaces accurately and prunes many impractical attacks to generate input payloads for vulnerability testing. To streamline vulnerability validation, PathSentinel leverages large language models (LLMs) to generate targeted exploit code based on the identified vulnerabilities and generated input payloads. The LLMs serve as a tool to reduce the engineering effort required for writing test applications, demonstrating the potential of combining static analysis with LLMs to enhance the efficiency of exploit generation and vulnerability validation. Evaluation on Android 12 and 14 systems from Samsung and OnePlus demonstrates PathSentinel's effectiveness, uncovering 51 previously unknown vulnerabilities among 217 apps with only 2 false positives. These results underscore the importance of combining program and access control policy analysis for accurate vulnerability detection and highlight the promising direction of integrating LLMs for automated exploit generation, providing a comprehensive approach to enhancing the security of Android systems against filesystem vulnerabilities.



## **15. Did the Neurons Read your Book? Document-level Membership Inference for Large Language Models**

cs.CL

Accepted at 33rd USENIX Security Symposium (USENIX Security 2024)

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2310.15007v2) [paper-pdf](http://arxiv.org/pdf/2310.15007v2)

**Authors**: Matthieu Meeus, Shubham Jain, Marek Rei, Yves-Alexandre de Montjoye

**Abstract**: With large language models (LLMs) poised to become embedded in our daily lives, questions are starting to be raised about the data they learned from. These questions range from potential bias or misinformation LLMs could retain from their training data to questions of copyright and fair use of human-generated text. However, while these questions emerge, developers of the recent state-of-the-art LLMs become increasingly reluctant to disclose details on their training corpus. We here introduce the task of document-level membership inference for real-world LLMs, i.e. inferring whether the LLM has seen a given document during training or not. First, we propose a procedure for the development and evaluation of document-level membership inference for LLMs by leveraging commonly used data sources for training and the model release date. We then propose a practical, black-box method to predict document-level membership and instantiate it on OpenLLaMA-7B with both books and academic papers. We show our methodology to perform very well, reaching an AUC of 0.856 for books and 0.678 for papers. We then show our approach to outperform the sentence-level membership inference attacks used in the privacy literature for the document-level membership task. We further evaluate whether smaller models might be less sensitive to document-level inference and show OpenLLaMA-3B to be approximately as sensitive as OpenLLaMA-7B to our approach. Finally, we consider two mitigation strategies and find the AUC to slowly decrease when only partial documents are considered but to remain fairly high when the model precision is reduced. Taken together, our results show that accurate document-level membership can be inferred for LLMs, increasing the transparency of technology poised to change our lives.



## **16. SLIP: Securing LLMs IP Using Weights Decomposition**

cs.CR

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.10886v1) [paper-pdf](http://arxiv.org/pdf/2407.10886v1)

**Authors**: Yehonathan Refael, Adam Hakim, Lev Greenberg, Tal Aviv, Satya Lokam, Ben Fishman, Shachar Seidman

**Abstract**: Large language models (LLMs) have recently seen widespread adoption, in both academia and industry. As these models grow, they become valuable intellectual property (IP), reflecting enormous investments by their owners. Moreover, the high cost of cloud-based deployment has driven interest towards deployment to edge devices, yet this risks exposing valuable parameters to theft and unauthorized use. Current methods to protect models' IP on the edge have limitations in terms of practicality, loss in accuracy, or suitability to requirements. In this paper, we introduce a novel hybrid inference algorithm, named SLIP, designed to protect edge-deployed models from theft. SLIP is the first hybrid protocol that is both practical for real-world applications and provably secure, while having zero accuracy degradation and minimal impact on latency. It involves partitioning the model between two computing resources, one secure but expensive, and another cost-effective but vulnerable. This is achieved through matrix decomposition, ensuring that the secure resource retains a maximally sensitive portion of the model's IP while performing a minimal amount of computations, and vice versa for the vulnerable resource. Importantly, the protocol includes security guarantees that prevent attackers from exploiting the partition to infer the secured information. Finally, we present experimental results that show the robustness and effectiveness of our method, positioning it as a compelling solution for protecting LLMs.



## **17. TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs**

cs.CR

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.09164v2) [paper-pdf](http://arxiv.org/pdf/2407.09164v2)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully used to simplify and facilitate code programming. With these tools, developers can easily generate desired complete functional codes based on incomplete code and natural language prompts. However, a few pioneering works revealed that these Code LLMs are also vulnerable, e.g., against backdoor and adversarial attacks. The former could induce LLMs to respond to triggers to insert malicious code snippets by poisoning the training data or model parameters, while the latter can craft malicious adversarial input codes to reduce the quality of generated codes. However, both attack methods have underlying limitations: backdoor attacks rely on controlling the model training process, while adversarial attacks struggle with fulfilling specific malicious purposes.   To inherit the advantages of both backdoor and adversarial attacks, this paper proposes a new attack paradigm, i.e., target-specific and adversarial prompt injection (TAPI), against Code LLMs. TAPI generates unreadable comments containing information about malicious instructions and hides them as triggers in the external source code. When users exploit Code LLMs to complete codes containing the trigger, the models will generate attacker-specified malicious code snippets at specific locations. We evaluate our TAPI attack on four representative LLMs under three representative malicious objectives and seven cases. The results show that our method is highly threatening (achieving an attack success rate enhancement of up to 89.3%) and stealthy (saving an average of 53.1% of tokens in the trigger design). In particular, we successfully attack some famous deployed code completion integrated applications, including CodeGeex and Github Copilot. This further confirms the realistic threat of our attack.



## **18. Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation**

cs.CV

ECCV2024 (Project Page: https://gyhdog99.github.io/projects/ecso/)

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2403.09572v3) [paper-pdf](http://arxiv.org/pdf/2403.09572v3)

**Authors**: Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T. Kwok, Yu Zhang

**Abstract**: Multimodal large language models (MLLMs) have shown impressive reasoning abilities. However, they are also more vulnerable to jailbreak attacks than their LLM predecessors. Although still capable of detecting the unsafe responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs can be easily bypassed with the introduction of image features. To construct robust MLLMs, we propose ECSO (Eyes Closed, Safety On), a novel training-free protecting approach that exploits the inherent safety awareness of MLLMs, and generates safer responses via adaptively transforming unsafe images into texts to activate the intrinsic safety mechanism of pre-aligned LLMs in MLLMs. Experiments on five state-of-the-art (SoTA) MLLMs demonstrate that ECSO enhances model safety significantly (e.g.,, 37.6% improvement on the MM-SafetyBench (SD+OCR) and 71.3% on VLSafe with LLaVA-1.5-7B), while consistently maintaining utility results on common MLLM benchmarks. Furthermore, we show that ECSO can be used as a data engine to generate supervised-finetuning (SFT) data for MLLM alignment without extra human intervention.



## **19. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

cs.CR

Found software bug in experiments, withdrawing in order to address  and update results

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2402.17012v4) [paper-pdf](http://arxiv.org/pdf/2402.17012v4)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model by leveraging recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Our code is available at github.com/safr-ai-lab/pandora-llm.



## **20. Merging Improves Self-Critique Against Jailbreak Attacks**

cs.CL

Published at ICML 2024 Workshop on Foundation Models in the Wild

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2406.07188v2) [paper-pdf](http://arxiv.org/pdf/2406.07188v2)

**Authors**: Victor Gallego

**Abstract**: The robustness of large language models (LLMs) against adversarial manipulations, such as jailbreak attacks, remains a significant challenge. In this work, we propose an approach that enhances the self-critique capability of the LLM and further fine-tunes it over sanitized synthetic data. This is done with the addition of an external critic model that can be merged with the original, thus bolstering self-critique capabilities and improving the robustness of the LLMs response to adversarial prompts. Our results demonstrate that the combination of merging and self-critique can reduce the attack success rate of adversaries significantly, thus offering a promising defense mechanism against jailbreak attacks. Code, data and models released at https://github.com/vicgalle/merging-self-critique-jailbreaks .



## **21. Watermarking Text Data on Large Language Models for Dataset Copyright**

cs.CR

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2305.13257v4) [paper-pdf](http://arxiv.org/pdf/2305.13257v4)

**Authors**: Yixin Liu, Hongsheng Hu, Xun Chen, Xuyun Zhang, Lichao Sun

**Abstract**: Substantial research works have shown that deep models, e.g., pre-trained models, on the large corpus can learn universal language representations, which are beneficial for downstream NLP tasks. However, these powerful models are also vulnerable to various privacy attacks, while much sensitive information exists in the training dataset. The attacker can easily steal sensitive information from public models, e.g., individuals' email addresses and phone numbers. In an attempt to address these issues, particularly the unauthorized use of private data, we introduce a novel watermarking technique via a backdoor-based membership inference approach named TextMarker, which can safeguard diverse forms of private information embedded in the training text data. Specifically, TextMarker only requires data owners to mark a small number of samples for data copyright protection under the black-box access assumption to the target model. Through extensive evaluation, we demonstrate the effectiveness of TextMarker on various real-world datasets, e.g., marking only 0.1% of the training dataset is practically sufficient for effective membership inference with negligible effect on model utility. We also discuss potential countermeasures and show that TextMarker is stealthy enough to bypass them.



## **22. Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training**

cs.CL

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.09121v1) [paper-pdf](http://arxiv.org/pdf/2407.09121v1)

**Authors**: Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Jiahao Xu, Tian Liang, Pinjia He, Zhaopeng Tu

**Abstract**: This study addresses a critical gap in safety tuning practices for Large Language Models (LLMs) by identifying and tackling a refusal position bias within safety tuning data, which compromises the models' ability to appropriately refuse generating unsafe content. We introduce a novel approach, Decoupled Refusal Training (DeRTa), designed to empower LLMs to refuse compliance to harmful prompts at any response position, significantly enhancing their safety capabilities. DeRTa incorporates two novel components: (1) Maximum Likelihood Estimation (MLE) with Harmful Response Prefix, which trains models to recognize and avoid unsafe content by appending a segment of harmful response to the beginning of a safe response, and (2) Reinforced Transition Optimization (RTO), which equips models with the ability to transition from potential harm to safety refusal consistently throughout the harmful response sequence. Our empirical evaluation, conducted using LLaMA3 and Mistral model families across six attack scenarios, demonstrates that our method not only improves model safety without compromising performance but also surpasses well-known models such as GPT-4 in defending against attacks. Importantly, our approach successfully defends recent advanced attack methods (e.g., CodeAttack) that have jailbroken GPT-4 and LLaMA3-70B-Instruct. Our code and data can be found at https://github.com/RobustNLP/DeRTa.



## **23. Jailbreaking as a Reward Misspecification Problem**

cs.LG

github url added

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2406.14393v2) [paper-pdf](http://arxiv.org/pdf/2406.14393v2)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts against various target aligned LLMs. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark while preserving the human readability of the generated prompts. Detailed analysis highlights the unique advantages brought by the proposed reward misspecification objective compared to previous methods.



## **24. A Survey of Attacks on Large Vision-Language Models: Resources, Advances, and Future Trends**

cs.CV

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.07403v2) [paper-pdf](http://arxiv.org/pdf/2407.07403v2)

**Authors**: Daizong Liu, Mingyu Yang, Xiaoye Qu, Pan Zhou, Yu Cheng, Wei Hu

**Abstract**: With the significant development of large models in recent years, Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across a wide range of multimodal understanding and reasoning tasks. Compared to traditional Large Language Models (LLMs), LVLMs present great potential and challenges due to its closer proximity to the multi-resource real-world applications and the complexity of multi-modal processing. However, the vulnerability of LVLMs is relatively underexplored, posing potential security risks in daily usage. In this paper, we provide a comprehensive review of the various forms of existing LVLM attacks. Specifically, we first introduce the background of attacks targeting LVLMs, including the attack preliminary, attack challenges, and attack resources. Then, we systematically review the development of LVLM attack methods, such as adversarial attacks that manipulate model outputs, jailbreak attacks that exploit model vulnerabilities for unauthorized actions, prompt injection attacks that engineer the prompt type and pattern, and data poisoning that affects model training. Finally, we discuss promising research directions in the future. We believe that our survey provides insights into the current landscape of LVLM vulnerabilities, inspiring more researchers to explore and mitigate potential safety issues in LVLM developments. The latest papers on LVLM attacks are continuously collected in https://github.com/liudaizong/Awesome-LVLM-Attack.



## **25. Tactics, Techniques, and Procedures (TTPs) in Interpreted Malware: A Zero-Shot Generation with Large Language Models**

cs.CR

19 pages, 11 figures

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08532v1) [paper-pdf](http://arxiv.org/pdf/2407.08532v1)

**Authors**: Ying Zhang, Xiaoyan Zhou, Hui Wen, Wenjia Niu, Jiqiang Liu, Haining Wang, Qiang Li

**Abstract**: Nowadays, the open-source software (OSS) ecosystem suffers from security threats of software supply chain (SSC) attacks. Interpreted OSS malware plays a vital role in SSC attacks, as criminals have an arsenal of attack vectors to deceive users into installing malware and executing malicious activities. In this paper, we introduce tactics, techniques, and procedures (TTPs) proposed by MITRE ATT\&CK into the interpreted malware analysis to characterize different phases of an attack lifecycle. Specifically, we propose GENTTP, a zero-shot approach to extracting a TTP of an interpreted malware package. GENTTP leverages large language models (LLMs) to automatically generate a TTP, where the input is a malicious package, and the output is a deceptive tactic and an execution tactic of attack vectors. To validate the effectiveness of GENTTP, we collect two datasets for evaluation: a dataset with ground truth labels and a large dataset in the wild. Experimental results show that GENTTP can generate TTPs with high accuracy and efficiency. To demonstrate GENTTP's benefits, we build an LLM-based Chatbot from 3,700+ PyPI malware's TTPs. We further conduct a quantitative analysis of malware's TTPs at a large scale. Our main findings include: (1) many OSS malicious packages share a relatively stable TTP, even with the increasing emergence of malware and attack campaigns, (2) a TTP reflects characteristics of a malware-based attack, and (3) an attacker's intent behind the malware is linked to a TTP.



## **26. Virtual Context: Enhancing Jailbreak Attacks with Special Token Injection**

cs.CR

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2406.19845v2) [paper-pdf](http://arxiv.org/pdf/2406.19845v2)

**Authors**: Yuqi Zhou, Lin Lu, Hanchi Sun, Pan Zhou, Lichao Sun

**Abstract**: Jailbreak attacks on large language models (LLMs) involve inducing these models to generate harmful content that violates ethics or laws, posing a significant threat to LLM security. Current jailbreak attacks face two main challenges: low success rates due to defensive measures and high resource requirements for crafting specific prompts. This paper introduces Virtual Context, which leverages special tokens, previously overlooked in LLM security, to improve jailbreak attacks. Virtual Context addresses these challenges by significantly increasing the success rates of existing jailbreak methods and requiring minimal background knowledge about the target model, thus enhancing effectiveness in black-box settings without additional overhead. Comprehensive evaluations show that Virtual Context-assisted jailbreak attacks can improve the success rates of four widely used jailbreak methods by approximately 40% across various LLMs. Additionally, applying Virtual Context to original malicious behaviors still achieves a notable jailbreak effect. In summary, our research highlights the potential of special tokens in jailbreak attacks and recommends including this threat in red-teaming testing to comprehensively enhance LLM security.



## **27. A Comprehensive Survey on the Security of Smart Grid: Challenges, Mitigations, and Future Research Opportunities**

cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07966v1) [paper-pdf](http://arxiv.org/pdf/2407.07966v1)

**Authors**: Arastoo Zibaeirad, Farnoosh Koleini, Shengping Bi, Tao Hou, Tao Wang

**Abstract**: In this study, we conduct a comprehensive review of smart grid security, exploring system architectures, attack methodologies, defense strategies, and future research opportunities. We provide an in-depth analysis of various attack vectors, focusing on new attack surfaces introduced by advanced components in smart grids. The review particularly includes an extensive analysis of coordinated attacks that incorporate multiple attack strategies and exploit vulnerabilities across various smart grid components to increase their adverse impact, demonstrating the complexity and potential severity of these threats. Following this, we examine innovative detection and mitigation strategies, including game theory, graph theory, blockchain, and machine learning, discussing their advancements in counteracting evolving threats and associated research challenges. In particular, our review covers a thorough examination of widely used machine learning-based mitigation strategies, analyzing their applications and research challenges spanning across supervised, unsupervised, semi-supervised, ensemble, and reinforcement learning. Further, we outline future research directions and explore new techniques and concerns. We first discuss the research opportunities for existing and emerging strategies, and then explore the potential role of new techniques, such as large language models (LLMs), and the emerging threat of adversarial machine learning in the future of smart grid security.



## **28. Flooding Spread of Manipulated Knowledge in LLM-Based Multi-Agent Communities**

cs.CL

18 Pages, working in progress

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07791v1) [paper-pdf](http://arxiv.org/pdf/2407.07791v1)

**Authors**: Tianjie Ju, Yiting Wang, Xinbei Ma, Pengzhou Cheng, Haodong Zhao, Yulong Wang, Lifeng Liu, Jian Xie, Zhuosheng Zhang, Gongshen Liu

**Abstract**: The rapid adoption of large language models (LLMs) in multi-agent systems has highlighted their impressive capabilities in various applications, such as collaborative problem-solving and autonomous negotiation. However, the security implications of these LLM-based multi-agent systems have not been thoroughly investigated, particularly concerning the spread of manipulated knowledge. In this paper, we investigate this critical issue by constructing a detailed threat model and a comprehensive simulation environment that mirrors real-world multi-agent deployments in a trusted platform. Subsequently, we propose a novel two-stage attack method involving Persuasiveness Injection and Manipulated Knowledge Injection to systematically explore the potential for manipulated knowledge (i.e., counterfactual and toxic knowledge) spread without explicit prompt manipulation.   Our method leverages the inherent vulnerabilities of LLMs in handling world knowledge, which can be exploited by attackers to unconsciously spread fabricated information. Through extensive experiments, we demonstrate that our attack method can successfully induce LLM-based agents to spread both counterfactual and toxic knowledge without degrading their foundational capabilities during agent communication. Furthermore, we show that these manipulations can persist through popular retrieval-augmented generation frameworks, where several benign agents store and retrieve manipulated chat histories for future interactions. This persistence indicates that even after the interaction has ended, the benign agents may continue to be influenced by manipulated knowledge. Our findings reveal significant security risks in LLM-based multi-agent systems, emphasizing the imperative need for robust defenses against manipulated knowledge spread, such as introducing ``guardian'' agents and advanced fact-checking tools.



## **29. SecureReg: Combining NLP and MLP for Enhanced Detection of Malicious Domain Name Registrations**

cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2401.03196v3) [paper-pdf](http://arxiv.org/pdf/2401.03196v3)

**Authors**: Furkan Çolhak, Mert İlhan Ecevit, Hasan Dağ, Reiner Creutzburg

**Abstract**: The escalating landscape of cyber threats, characterized by the registration of thousands of new domains daily for large-scale Internet attacks such as spam, phishing, and drive-by downloads, underscores the imperative for innovative detection methodologies. This paper introduces a cutting-edge approach for identifying suspicious domains at the onset of the registration process. The accompanying data pipeline generates crucial features by comparing new domains to registered domains, emphasizing the crucial similarity score. The proposed system analyzes semantic and numerical attributes by leveraging a novel combination of Natural Language Processing (NLP) techniques, including a pretrained CANINE model and Multilayer Perceptron (MLP) models, providing a robust solution for early threat detection. This integrated Pretrained NLP (CANINE) + MLP model showcases the outstanding performance, surpassing both individual pretrained NLP models and standalone MLP models. With an F1 score of 84.86\% and an accuracy of 84.95\% on the SecureReg dataset, it effectively detects malicious domain registrations. The findings demonstrate the effectiveness of the integrated approach and contribute to the ongoing efforts to develop proactive strategies to mitigate the risks associated with illicit online activities through the early identification of suspicious domain registrations.



## **30. Evaluating the Adversarial Robustness of Retrieval-Based In-Context Learning for Large Language Models**

cs.CL

COLM 2024, 29 pages, 6 figures

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2405.15984v2) [paper-pdf](http://arxiv.org/pdf/2405.15984v2)

**Authors**: Simon Chi Lok Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl



## **31. Was it Slander? Towards Exact Inversion of Generative Language Models**

cs.CR

4 pages, 3 figures

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.11059v1) [paper-pdf](http://arxiv.org/pdf/2407.11059v1)

**Authors**: Adrians Skapars, Edoardo Manino, Youcheng Sun, Lucas C. Cordeiro

**Abstract**: Training large language models (LLMs) requires a substantial investment of time and money. To get a good return on investment, the developers spend considerable effort ensuring that the model never produces harmful and offensive outputs. However, bad-faith actors may still try to slander the reputation of an LLM by publicly reporting a forged output. In this paper, we show that defending against such slander attacks requires reconstructing the input of the forged output or proving that it does not exist. To do so, we propose and evaluate a search based approach for targeted adversarial attacks for LLMs. Our experiments show that we are rarely able to reconstruct the exact input of an arbitrary output, thus demonstrating that LLMs are still vulnerable to slander attacks.



## **32. The Ethics of Interaction: Mitigating Security Threats in LLMs**

cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2401.12273v2) [paper-pdf](http://arxiv.org/pdf/2401.12273v2)

**Authors**: Ashutosh Kumar, Shiv Vignesh Murthy, Sagarika Singh, Swathy Ragupathy

**Abstract**: This paper comprehensively explores the ethical challenges arising from security threats to Large Language Models (LLMs). These intricate digital repositories are increasingly integrated into our daily lives, making them prime targets for attacks that can compromise their training data and the confidentiality of their data sources. The paper delves into the nuanced ethical repercussions of such security threats on society and individual privacy. We scrutinize five major threats--prompt injection, jailbreaking, Personal Identifiable Information (PII) exposure, sexually explicit content, and hate-based content--going beyond mere identification to assess their critical ethical consequences and the urgency they create for robust defensive strategies. The escalating reliance on LLMs underscores the crucial need for ensuring these systems operate within the bounds of ethical norms, particularly as their misuse can lead to significant societal and individual harm. We propose conceptualizing and developing an evaluative tool tailored for LLMs, which would serve a dual purpose: guiding developers and designers in preemptive fortification of backend systems and scrutinizing the ethical dimensions of LLM chatbot responses during the testing phase. By comparing LLM responses with those expected from humans in a moral context, we aim to discern the degree to which AI behaviors align with the ethical values held by a broader society. Ultimately, this paper not only underscores the ethical troubles presented by LLMs; it also highlights a path toward cultivating trust in these systems.



## **33. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

cs.IR

Survey paper

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06992v1) [paper-pdf](http://arxiv.org/pdf/2407.06992v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.



## **34. A hybrid LLM workflow can help identify user privilege related variables in programs of any size**

cs.CR

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2403.15723v2) [paper-pdf](http://arxiv.org/pdf/2403.15723v2)

**Authors**: Haizhou Wang, Zhilong Wang, Peng Liu

**Abstract**: Many programs involves operations and logic manipulating user privileges, which is essential for the security of an organization. Therefore, one common malicious goal of attackers is to obtain or escalate the privileges, causing privilege leakage. To protect the program and the organization against privilege leakage attacks, it is important to eliminate the vulnerabilities which can be exploited to achieve such attacks. Unfortunately, while memory vulnerabilities are less challenging to find, logic vulnerabilities are much more imminent, harmful and difficult to identify. Accordingly, many analysts choose to find user privilege related (UPR) variables first as start points to investigate the code where the UPR variables may be used to see if there exists any vulnerabilities, especially the logic ones. In this paper, we introduce a large language model (LLM) workflow that can assist analysts in identifying such UPR variables, which is considered to be a very time-consuming task. Specifically, our tool will audit all the variables in a program and output a UPR score, which is the degree of relationship (closeness) between the variable and user privileges, for each variable. The proposed approach avoids the drawbacks introduced by directly prompting a LLM to find UPR variables by focusing on leverage the LLM at statement level instead of supplying LLM with very long code snippets. Those variables with high UPR scores are essentially potential UPR variables, which should be manually investigated. Our experiments show that using a typical UPR score threshold (i.e., UPR score >0.8), the false positive rate (FPR) is only 13.49%, while UPR variable found is significantly more than that of the heuristic based method.



## **35. Does CLIP Know My Face?**

cs.LG

Published in the Journal of Artificial Intelligence Research (JAIR)

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2209.07341v4) [paper-pdf](http://arxiv.org/pdf/2209.07341v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Manuel Brack, Felix Friedrich, Patrick Schramowski, Kristian Kersting

**Abstract**: With the rise of deep learning in various applications, privacy concerns around the protection of training data have become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for stronger privacy protection in large-scale models and suggest that IDIAs can be used to prove the unauthorized use of data for training and to enforce privacy laws.



## **36. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

cs.CR

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2406.03230v3) [paper-pdf](http://arxiv.org/pdf/2406.03230v3)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.



## **37. Exposing Privacy Gaps: Membership Inference Attack on Preference Data for LLM Alignment**

cs.AI

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.06443v1) [paper-pdf](http://arxiv.org/pdf/2407.06443v1)

**Authors**: Qizhang Feng, Siva Rajesh Kasa, Hyokun Yun, Choon Hui Teo, Sravan Babu Bodapati

**Abstract**: Large Language Models (LLMs) have seen widespread adoption due to their remarkable natural language capabilities. However, when deploying them in real-world settings, it is important to align LLMs to generate texts according to acceptable human standards. Methods such as Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO) have made significant progress in refining LLMs using human preference data. However, the privacy concerns inherent in utilizing such preference data have yet to be adequately studied. In this paper, we investigate the vulnerability of LLMs aligned using human preference datasets to membership inference attacks (MIAs), highlighting the shortcomings of previous MIA approaches with respect to preference data. Our study has two main contributions: first, we introduce a novel reference-based attack framework specifically for analyzing preference data called PREMIA (\uline{Pre}ference data \uline{MIA}); second, we provide empirical evidence that DPO models are more vulnerable to MIA compared to PPO models. Our findings highlight gaps in current privacy-preserving practices for LLM alignment.



## **38. If You Don't Understand It, Don't Use It: Eliminating Trojans with Filters Between Layers**

cs.LG

11 pages, 6 figures

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.06411v1) [paper-pdf](http://arxiv.org/pdf/2407.06411v1)

**Authors**: Adriano Hernandez

**Abstract**: Large language models (LLMs) sometimes exhibit dangerous unintended behaviors. Finding and fixing these is challenging because the attack surface is massive -- it is not tractable to exhaustively search for all possible inputs that may elicit such behavior. One specific and particularly challenging case is that if data-poisoning-injected trojans, since there is no way to know what they are to search for them. To our knowledge, there is no generally applicable method to unlearn unknown trojans injected during pre-training. This work seeks to provide a general purpose recipe (filters) and a specific implementation (LoRA) filters that work in practice on small to medium sized models. The focus is primarily empirical, though some perplexing behavior opens the door to the fundamental question of how LLMs store and process information. Not unexpectedly, we find that our filters work best on the residual stream and the latest layers.



## **39. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

cs.LG

Code available at https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2401.17263v4) [paper-pdf](http://arxiv.org/pdf/2401.17263v4)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo



## **40. Adaptive and robust watermark against model extraction attack**

cs.CR

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2405.02365v2) [paper-pdf](http://arxiv.org/pdf/2405.02365v2)

**Authors**: Kaiyi Pang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.



## **41. Exploring the Adversarial Capabilities of Large Language Models**

cs.AI

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2402.09132v4) [paper-pdf](http://arxiv.org/pdf/2402.09132v4)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.



## **42. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

cs.CR

19 pages, 14 figures, 4 tables

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2405.13401v4) [paper-pdf](http://arxiv.org/pdf/2405.13401v4)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.



## **43. BadCLM: Backdoor Attack in Clinical Language Models for Electronic Health Records**

cs.CL

AMIA 2024

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.05213v1) [paper-pdf](http://arxiv.org/pdf/2407.05213v1)

**Authors**: Weimin Lyu, Zexin Bi, Fusheng Wang, Chao Chen

**Abstract**: The advent of clinical language models integrated into electronic health records (EHR) for clinical decision support has marked a significant advancement, leveraging the depth of clinical notes for improved decision-making. Despite their success, the potential vulnerabilities of these models remain largely unexplored. This paper delves into the realm of backdoor attacks on clinical language models, introducing an innovative attention-based backdoor attack method, BadCLM (Bad Clinical Language Models). This technique clandestinely embeds a backdoor within the models, causing them to produce incorrect predictions when a pre-defined trigger is present in inputs, while functioning accurately otherwise. We demonstrate the efficacy of BadCLM through an in-hospital mortality prediction task with MIMIC III dataset, showcasing its potential to compromise model integrity. Our findings illuminate a significant security risk in clinical decision support systems and pave the way for future endeavors in fortifying clinical language models against such vulnerabilities.



## **44. LLMCloudHunter: Harnessing LLMs for Automated Extraction of Detection Rules from Cloud-Based CTI**

cs.CR

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.05194v1) [paper-pdf](http://arxiv.org/pdf/2407.05194v1)

**Authors**: Yuval Schwartz, Lavi Benshimol, Dudu Mimran, Yuval Elovici, Asaf Shabtai

**Abstract**: As the number and sophistication of cyber attacks have increased, threat hunting has become a critical aspect of active security, enabling proactive detection and mitigation of threats before they cause significant harm. Open-source cyber threat intelligence (OS-CTI) is a valuable resource for threat hunters, however, it often comes in unstructured formats that require further manual analysis. Previous studies aimed at automating OSCTI analysis are limited since (1) they failed to provide actionable outputs, (2) they did not take advantage of images present in OSCTI sources, and (3) they focused on on-premises environments, overlooking the growing importance of cloud environments. To address these gaps, we propose LLMCloudHunter, a novel framework that leverages large language models (LLMs) to automatically generate generic-signature detection rule candidates from textual and visual OSCTI data. We evaluated the quality of the rules generated by the proposed framework using 12 annotated real-world cloud threat reports. The results show that our framework achieved a precision of 92% and recall of 98% for the task of accurately extracting API calls made by the threat actor and a precision of 99% with a recall of 98% for IoCs. Additionally, 99.18% of the generated detection rule candidates were successfully compiled and converted into Splunk queries.



## **45. On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks**

cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04794v1) [paper-pdf](http://arxiv.org/pdf/2407.04794v1)

**Authors**: Zesen Liu, Tianshuo Cong, Xinlei He, Qi Li

**Abstract**: Large Language Models (LLMs) excel in various applications, including text generation and complex tasks. However, the misuse of LLMs raises concerns about the authenticity and ethical implications of the content they produce, such as deepfake news, academic fraud, and copyright infringement. Watermarking techniques, which embed identifiable markers in machine-generated text, offer a promising solution to these issues by allowing for content verification and origin tracing. Unfortunately, the robustness of current LLM watermarking schemes under potential watermark removal attacks has not been comprehensively explored.   In this paper, to fill this gap, we first systematically comb the mainstream watermarking schemes and removal attacks on machine-generated texts, and then we categorize them into pre-text (before text generation) and post-text (after text generation) classes so that we can conduct diversified analyses. In our experiments, we evaluate eight watermarks (five pre-text, three post-text) and twelve attacks (two pre-text, ten post-text) across 87 scenarios. Evaluation results indicate that (1) KGW and Exponential watermarks offer high text quality and watermark retention but remain vulnerable to most attacks; (2) Post-text attacks are found to be more efficient and practical than pre-text attacks; (3) Pre-text watermarks are generally more imperceptible, as they do not alter text fluency, unlike post-text watermarks; (4) Additionally, combined attack methods can significantly increase effectiveness, highlighting the need for more robust watermarking solutions. Our study underscores the vulnerabilities of current techniques and the necessity for developing more resilient schemes.



## **46. Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models**

cs.SD

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04482v1) [paper-pdf](http://arxiv.org/pdf/2407.04482v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Speech enabled foundation models, either in the form of flexible speech recognition based systems or audio-prompted large language models (LLMs), are becoming increasingly popular. One of the interesting aspects of these models is their ability to perform tasks other than automatic speech recognition (ASR) using an appropriate prompt. For example, the OpenAI Whisper model can perform both speech transcription and speech translation. With the development of audio-prompted LLMs there is the potential for even greater control options. In this work we demonstrate that with this greater flexibility the systems can be susceptible to model-control adversarial attacks. Without any access to the model prompt it is possible to modify the behaviour of the system by appropriately changing the audio input. To illustrate this risk, we demonstrate that it is possible to prepend a short universal adversarial acoustic segment to any input speech signal to override the prompt setting of an ASR foundation model. Specifically, we successfully use a universal adversarial acoustic segment to control Whisper to always perform speech translation, despite being set to perform speech transcription. Overall, this work demonstrates a new form of adversarial attack on multi-tasking speech enabled foundation models that needs to be considered prior to the deployment of this form of model.



## **47. Waterfall: Framework for Robust and Scalable Text Watermarking**

cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04411v1) [paper-pdf](http://arxiv.org/pdf/2407.04411v1)

**Authors**: Gregory Kang Ruey Lau, Xinyuan Niu, Hieu Dao, Jiangwei Chen, Chuan-Sheng Foo, Bryan Kian Hsiang Low

**Abstract**: Protecting intellectual property (IP) of text such as articles and code is increasingly important, especially as sophisticated attacks become possible, such as paraphrasing by large language models (LLMs) or even unauthorized training of LLMs on copyrighted text to infringe such IP. However, existing text watermarking methods are not robust enough against such attacks nor scalable to millions of users for practical implementation. In this paper, we propose Waterfall, the first training-free framework for robust and scalable text watermarking applicable across multiple text types (e.g., articles, code) and languages supportable by LLMs, for general text and LLM data provenance. Waterfall comprises several key innovations, such as being the first to use LLM as paraphrasers for watermarking along with a novel combination of techniques that are surprisingly effective in achieving robust verifiability and scalability. We empirically demonstrate that Waterfall achieves significantly better scalability, robust verifiability, and computational efficiency compared to SOTA article-text watermarking methods, and also showed how it could be directly applied to the watermarking of code.



## **48. Jailbreak Attacks and Defenses Against Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04295v1) [paper-pdf](http://arxiv.org/pdf/2407.04295v1)

**Authors**: Sibo Yi, Yule Liu, Zhen Sun, Tianshuo Cong, Xinlei He, Jiaxing Song, Ke Xu, Qi Li

**Abstract**: Large Language Models (LLMs) have performed exceptionally in various text-generative tasks, including question answering, translation, code completion, etc. However, the over-assistance of LLMs has raised the challenge of "jailbreaking", which induces the model to generate malicious responses against the usage policy and society by designing adversarial prompts. With the emergence of jailbreak attack methods exploiting different vulnerabilities in LLMs, the corresponding safety alignment measures are also evolving. In this paper, we propose a comprehensive and detailed taxonomy of jailbreak attack and defense methods. For instance, the attack methods are divided into black-box and white-box attacks based on the transparency of the target model. Meanwhile, we classify defense methods into prompt-level and model-level defenses. Additionally, we further subdivide these attack and defense methods into distinct sub-classes and present a coherent diagram illustrating their relationships. We also conduct an investigation into the current evaluation methods and compare them from different perspectives. Our findings aim to inspire future research and practical implementations in safeguarding LLMs against adversarial attacks. Above all, although jailbreak remains a significant concern within the community, we believe that our work enhances the understanding of this domain and provides a foundation for developing more secure LLMs.



## **49. Defending Jailbreak Prompts via In-Context Adversarial Game**

cs.LG

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2402.13148v2) [paper-pdf](http://arxiv.org/pdf/2402.13148v2)

**Authors**: Yujun Zhou, Yufei Han, Haomin Zhuang, Kehan Guo, Zhenwen Liang, Hongyan Bao, Xiangliang Zhang

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Moreover, ICAG demonstrates remarkable transferability to other LLMs, indicating its potential as a versatile defense mechanism.



## **50. Defense Against Syntactic Textual Backdoor Attacks with Token Substitution**

cs.CL

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.04179v1) [paper-pdf](http://arxiv.org/pdf/2407.04179v1)

**Authors**: Xinglin Li, Xianwen He, Yao Li, Minhao Cheng

**Abstract**: Textual backdoor attacks present a substantial security risk to Large Language Models (LLM). It embeds carefully chosen triggers into a victim model at the training stage, and makes the model erroneously predict inputs containing the same triggers as a certain class. Prior backdoor defense methods primarily target special token-based triggers, leaving syntax-based triggers insufficiently addressed. To fill this gap, this paper proposes a novel online defense algorithm that effectively counters syntax-based as well as special token-based backdoor attacks. The algorithm replaces semantically meaningful words in sentences with entirely different ones but preserves the syntactic templates or special tokens, and then compares the predicted labels before and after the substitution to determine whether a sentence contains triggers. Experimental results confirm the algorithm's performance against these two types of triggers, offering a comprehensive defense strategy for model integrity.



