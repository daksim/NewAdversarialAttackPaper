# Latest Large Language Model Attack Papers
**update at 2024-05-06 11:08:03**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2308.07308v4) [paper-pdf](http://arxiv.org/pdf/2308.07308v4)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2. The code is publicly available at https://github.com/poloclub/llm-self-defense



## **2. Boosting Jailbreak Attack with Momentum**

cs.LG

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01229v1) [paper-pdf](http://arxiv.org/pdf/2405.01229v1)

**Authors**: Yihao Zhang, Zeming Wei

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse tasks, yet they remain vulnerable to adversarial attacks, notably the well-documented \textit{jailbreak} attack. Recently, the Greedy Coordinate Gradient (GCG) attack has demonstrated efficacy in exploiting this vulnerability by optimizing adversarial prompts through a combination of gradient heuristics and greedy search. However, the efficiency of this attack has become a bottleneck in the attacking process. To mitigate this limitation, in this paper we rethink the generation of adversarial prompts through an optimization lens, aiming to stabilize the optimization process and harness more heuristic insights from previous iterations. Specifically, we introduce the \textbf{M}omentum \textbf{A}ccelerated G\textbf{C}G (\textbf{MAC}) attack, which incorporates a momentum term into the gradient heuristic. Experimental results showcase the notable enhancement achieved by MAP in gradient-based attacks on aligned language models. Our code is available at https://github.com/weizeming/momentum-attack-llm.



## **3. Adversarial Attacks and Defense for Conversation Entailment Task**

cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.00289v2) [paper-pdf](http://arxiv.org/pdf/2405.00289v2)

**Authors**: Zhenning Yang, Ryan Krawec, Liang-Yuan Wu

**Abstract**: As the deployment of NLP systems in critical applications grows, ensuring the robustness of large language models (LLMs) against adversarial attacks becomes increasingly important. Large language models excel in various NLP tasks but remain vulnerable to low-cost adversarial attacks. Focusing on the domain of conversation entailment, where multi-turn dialogues serve as premises to verify hypotheses, we fine-tune a transformer model to accurately discern the truthfulness of these hypotheses. Adversaries manipulate hypotheses through synonym swapping, aiming to deceive the model into making incorrect predictions. To counteract these attacks, we implemented innovative fine-tuning techniques and introduced an embedding perturbation loss method to significantly bolster the model's robustness. Our findings not only emphasize the importance of defending against adversarial attacks in NLP but also highlight the real-world implications, suggesting that enhancing model robustness is critical for reliable NLP applications.



## **4. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.07921v2) [paper-pdf](http://arxiv.org/pdf/2404.07921v2)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.



## **5. Assessing LLMs in Malicious Code Deobfuscation of Real-world Malware Campaigns**

cs.CR

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19715v1) [paper-pdf](http://arxiv.org/pdf/2404.19715v1)

**Authors**: Constantinos Patsakis, Fran Casino, Nikolaos Lykousas

**Abstract**: The integration of large language models (LLMs) into various pipelines is increasingly widespread, effectively automating many manual tasks and often surpassing human capabilities. Cybersecurity researchers and practitioners have recognised this potential. Thus, they are actively exploring its applications, given the vast volume of heterogeneous data that requires processing to identify anomalies, potential bypasses, attacks, and fraudulent incidents. On top of this, LLMs' advanced capabilities in generating functional code, comprehending code context, and summarising its operations can also be leveraged for reverse engineering and malware deobfuscation. To this end, we delve into the deobfuscation capabilities of state-of-the-art LLMs. Beyond merely discussing a hypothetical scenario, we evaluate four LLMs with real-world malicious scripts used in the notorious Emotet malware campaign. Our results indicate that while not absolutely accurate yet, some LLMs can efficiently deobfuscate such payloads. Thus, fine-tuning LLMs for this task can be a viable potential for future AI-powered threat intelligence pipelines in the fight against obfuscated malware.



## **6. Transferring Troubles: Cross-Lingual Transferability of Backdoor Attacks in LLMs with Instruction Tuning**

cs.CL

work in progress

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19597v1) [paper-pdf](http://arxiv.org/pdf/2404.19597v1)

**Authors**: Xuanli He, Jun Wang, Qiongkai Xu, Pasquale Minervini, Pontus Stenetorp, Benjamin I. P. Rubinstein, Trevor Cohn

**Abstract**: The implications of backdoor attacks on English-centric large language models (LLMs) have been widely examined - such attacks can be achieved by embedding malicious behaviors during training and activated under specific conditions that trigger malicious outputs. However, the impact of backdoor attacks on multilingual models remains under-explored. Our research focuses on cross-lingual backdoor attacks against multilingual LLMs, particularly investigating how poisoning the instruction-tuning data in one or two languages can affect the outputs in languages whose instruction-tuning data was not poisoned. Despite its simplicity, our empirical analysis reveals that our method exhibits remarkable efficacy in models like mT5, BLOOM, and GPT-3.5-turbo, with high attack success rates, surpassing 95% in several languages across various scenarios. Alarmingly, our findings also indicate that larger models show increased susceptibility to transferable cross-lingual backdoor attacks, which also applies to LLMs predominantly pre-trained on English data, such as Llama2, Llama3, and Gemma. Moreover, our experiments show that triggers can still work even after paraphrasing, and the backdoor mechanism proves highly effective in cross-lingual response settings across 25 languages, achieving an average attack success rate of 50%. Our study aims to highlight the vulnerabilities and significant security risks present in current multilingual LLMs, underscoring the emergent need for targeted security measures.



## **7. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

cs.CV

16 pages, 14 figures

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19287v1) [paper-pdf](http://arxiv.org/pdf/2404.19287v1)

**Authors**: Wanqi Zhou, Shuanghao Bai, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP have shown impressive generalization performance across various downstream tasks, yet they remain vulnerable to adversarial attacks. While prior research has primarily concentrated on improving the adversarial robustness of image encoders to guard against attacks on images, the exploration of text-based and multimodal attacks has largely been overlooked. In this work, we initiate the first known and comprehensive effort to study adapting vision-language models for adversarial robustness under the multimodal attack. Firstly, we introduce a multimodal attack strategy and investigate the impact of different attacks. We then propose a multimodal contrastive adversarial training loss, aligning the clean and adversarial text embeddings with the adversarial and clean visual features, to enhance the adversarial robustness of both image and text encoders of CLIP. Extensive experiments on 15 datasets across two tasks demonstrate that our method significantly improves the adversarial robustness of CLIP. Interestingly, we find that the model fine-tuned against multimodal adversarial attacks exhibits greater robustness than its counterpart fine-tuned solely against image-based attacks, even in the context of image attacks, which may open up new possibilities for enhancing the security of VLMs.



## **8. Intention Analysis Makes LLMs A Good Jailbreak Defender**

cs.CL

20 pages, 16 figures

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2401.06561v3) [paper-pdf](http://arxiv.org/pdf/2401.06561v3)

**Authors**: Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao

**Abstract**: Aligning large language models (LLMs) with human values, particularly in the face of complex and stealthy jailbreak attacks, presents a formidable challenge. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis ($\mathbb{IA}$). The principle behind this is to trigger LLMs' inherent self-correct and improve ability through a two-stage process: 1) essential intention analysis, and 2) policy-aligned response. Notably, $\mathbb{IA}$ is an inference-only method, thus could enhance the safety of LLMs without compromising their helpfulness. Extensive experiments on varying jailbreak benchmarks across ChatGLM, LLaMA2, Vicuna, MPT, DeepSeek, and GPT-3.5 show that $\mathbb{IA}$ could consistently and significantly reduce the harmfulness in responses (averagely -53.1% attack success rate) and maintain the general helpfulness. Encouragingly, with the help of our $\mathbb{IA}$, Vicuna-7B even outperforms GPT-3.5 in terms of attack success rate. Further analyses present some insights into how our method works. To facilitate reproducibility, we release our code and scripts at: https://github.com/alphadl/SafeLLM_with_IntentionAnalysis.



## **9. AppPoet: Large Language Model based Android malware detection via multi-view prompt engineering**

cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18816v1) [paper-pdf](http://arxiv.org/pdf/2404.18816v1)

**Authors**: Wenxiang Zhao, Juntao Wu, Zhaoyi Meng

**Abstract**: Due to the vast array of Android applications, their multifarious functions and intricate behavioral semantics, attackers can adopt various tactics to conceal their genuine attack intentions within legitimate functions. However, numerous feature engineering based methods suffer from a limitation in mining behavioral semantic information, thus impeding the accuracy and efficiency of Android malware detection. Besides, the majority of existing feature engineering based methods are weakly interpretive and fail to furnish researchers with effective and readable detection reports. Inspired by the success of the Large Language Models (LLMs) in natural language understanding, we propose AppPoet, a LLM-assisted multi-view system for Android malware detection. Firstly, AppPoet employs a static method to comprehensively collect application features and formulate various observation views. Subsequently, it steers the LLM to produce function descriptions and behavioral summaries for views via our meticulously devised multi-view prompt engineering technique to realize the deep mining of view semantics. Finally, we collaboratively fuse the multi-view information to efficiently and accurately detect malware through a deep neural network (DNN) classifier and then generate the heuristic diagnostic reports. Experimental results demonstrate that our method achieves a detection accuracy of 97.15% and an F1 score of 97.21%, which is superior to the baseline method Drebin and its variant. Furthermore, the case study evaluates the effectiveness of our generated diagnostic reports.



## **10. Universal Jailbreak Backdoors from Poisoned Human Feedback**

cs.AI

Accepted as conference paper in ICLR 2024

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2311.14455v4) [paper-pdf](http://arxiv.org/pdf/2311.14455v4)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.



## **11. Assessing Cybersecurity Vulnerabilities in Code Large Language Models**

cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18567v1) [paper-pdf](http://arxiv.org/pdf/2404.18567v1)

**Authors**: Md Imran Hossen, Jianyi Zhang, Yinzhi Cao, Xiali Hei

**Abstract**: Instruction-tuned Code Large Language Models (Code LLMs) are increasingly utilized as AI coding assistants and integrated into various applications. However, the cybersecurity vulnerabilities and implications arising from the widespread integration of these models are not yet fully understood due to limited research in this domain. To bridge this gap, this paper presents EvilInstructCoder, a framework specifically designed to assess the cybersecurity vulnerabilities of instruction-tuned Code LLMs to adversarial attacks. EvilInstructCoder introduces the Adversarial Code Injection Engine to automatically generate malicious code snippets and inject them into benign code to poison instruction tuning datasets. It incorporates practical threat models to reflect real-world adversaries with varying capabilities and evaluates the exploitability of instruction-tuned Code LLMs under these diverse adversarial attack scenarios. Through the use of EvilInstructCoder, we conduct a comprehensive investigation into the exploitability of instruction tuning for coding tasks using three state-of-the-art Code LLM models: CodeLlama, DeepSeek-Coder, and StarCoder2, under various adversarial attack scenarios. Our experimental results reveal a significant vulnerability in these models, demonstrating that adversaries can manipulate the models to generate malicious payloads within benign code contexts in response to natural language instructions. For instance, under the backdoor attack setting, by poisoning only 81 samples (0.5\% of the entire instruction dataset), we achieve Attack Success Rate at 1 (ASR@1) scores ranging from 76\% to 86\% for different model families. Our study sheds light on the critical cybersecurity vulnerabilities posed by instruction-tuned Code LLMs and emphasizes the urgent necessity for robust defense mechanisms to mitigate the identified vulnerabilities.



## **12. Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning**

cs.LG

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2401.10862v2) [paper-pdf](http://arxiv.org/pdf/2401.10862v2)

**Authors**: Adib Hasan, Ileana Rugina, Alex Wang

**Abstract**: Large Language Models (LLMs) are susceptible to `jailbreaking' prompts, which can induce the generation of harmful content. This paper demonstrates that moderate WANDA pruning (Sun et al., 2023) can increase their resistance to such attacks without the need for fine-tuning, while maintaining performance on standard benchmarks. Our findings suggest that the benefits of pruning correlate with the initial safety levels of the model, indicating a regularizing effect of WANDA pruning. We introduce a dataset of 225 harmful tasks across five categories to systematically evaluate this safety enhancement. We argue that safety improvements can be understood through a regularization perspective. First, we show that pruning helps LLMs focus more effectively on task-relevant tokens within jailbreaking prompts. Then, we analyze the effects of pruning on the perplexity of malicious prompts before and after their integration into jailbreak templates. Finally, we demonstrate statistically significant performance improvements under domain shifts when applying WANDA to linear models.



## **13. Learnable Linguistic Watermarks for Tracing Model Extraction Attacks on Large Language Models**

cs.CR

not decided

**SubmitDate**: 2024-04-28    [abs](http://arxiv.org/abs/2405.01509v1) [paper-pdf](http://arxiv.org/pdf/2405.01509v1)

**Authors**: Minhao Bai, Kaiyi Pang, Yongfeng Huang

**Abstract**: In the rapidly evolving domain of artificial intelligence, safeguarding the intellectual property of Large Language Models (LLMs) is increasingly crucial. Current watermarking techniques against model extraction attacks, which rely on signal insertion in model logits or post-processing of generated text, remain largely heuristic. We propose a novel method for embedding learnable linguistic watermarks in LLMs, aimed at tracing and preventing model extraction attacks. Our approach subtly modifies the LLM's output distribution by introducing controlled noise into token frequency distributions, embedding an statistically identifiable controllable watermark.We leverage statistical hypothesis testing and information theory, particularly focusing on Kullback-Leibler Divergence, to differentiate between original and modified distributions effectively. Our watermarking method strikes a delicate well balance between robustness and output quality, maintaining low false positive/negative rates and preserving the LLM's original performance.



## **14. Investigating the prompt leakage effect and black-box defenses for multi-turn LLM interactions**

cs.CR

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2404.16251v2) [paper-pdf](http://arxiv.org/pdf/2404.16251v2)

**Authors**: Divyansh Agarwal, Alexander R. Fabbri, Philippe Laban, Ben Risher, Shafiq Joty, Caiming Xiong, Chien-Sheng Wu

**Abstract**: Prompt leakage in large language models (LLMs) poses a significant security and privacy threat, particularly in retrieval-augmented generation (RAG) systems. However, leakage in multi-turn LLM interactions along with mitigation strategies has not been studied in a standardized manner. This paper investigates LLM vulnerabilities against prompt leakage across 4 diverse domains and 10 closed- and open-source LLMs. Our unique multi-turn threat model leverages the LLM's sycophancy effect and our analysis dissects task instruction and knowledge leakage in the LLM response. In a multi-turn setting, our threat model elevates the average attack success rate (ASR) to 86.2%, including a 99% leakage with GPT-4 and claude-1.3. We find that some black-box LLMs like Gemini show variable susceptibility to leakage across domains - they are more likely to leak contextual knowledge in the news domain compared to the medical domain. Our experiments measure specific effects of 6 black-box defense strategies, including a query-rewriter in the RAG scenario. Our proposed multi-tier combination of defenses still has an ASR of 5.3% for black-box LLMs, indicating room for enhancement and future direction for LLM security research.



## **15. Don't Say No: Jailbreaking LLM by Suppressing Refusal**

cs.CL

**SubmitDate**: 2024-04-25    [abs](http://arxiv.org/abs/2404.16369v1) [paper-pdf](http://arxiv.org/pdf/2404.16369v1)

**Authors**: Yukai Zhou, Wenjie Wang

**Abstract**: Ensuring the safety alignment of Large Language Models (LLMs) is crucial to generating responses consistent with human values. Despite their ability to recognize and avoid harmful queries, LLMs are vulnerable to "jailbreaking" attacks, where carefully crafted prompts elicit them to produce toxic content. One category of jailbreak attacks is reformulating the task as adversarial attacks by eliciting the LLM to generate an affirmative response. However, the typical attack in this category GCG has very limited attack success rate. In this study, to better study the jailbreak attack, we introduce the DSN (Don't Say No) attack, which prompts LLMs to not only generate affirmative responses but also novelly enhance the objective to suppress refusals. In addition, another challenge lies in jailbreak attacks is the evaluation, as it is difficult to directly and accurately assess the harmfulness of the attack. The existing evaluation such as refusal keyword matching has its own limitation as it reveals numerous false positive and false negative instances. To overcome this challenge, we propose an ensemble evaluation pipeline incorporating Natural Language Inference (NLI) contradiction assessment and two external LLM evaluators. Extensive experiments demonstrate the potency of the DSN and the effectiveness of ensemble evaluation compared to baseline methods.



## **16. Attacks on Third-Party APIs of Large Language Models**

cs.CR

ICLR 2024 Workshop on Secure and Trustworthy Large Language Models

**SubmitDate**: 2024-04-24    [abs](http://arxiv.org/abs/2404.16891v1) [paper-pdf](http://arxiv.org/pdf/2404.16891v1)

**Authors**: Wanru Zhao, Vidit Khazanchi, Haodi Xing, Xuanli He, Qiongkai Xu, Nicholas Donald Lane

**Abstract**: Large language model (LLM) services have recently begun offering a plugin ecosystem to interact with third-party API services. This innovation enhances the capabilities of LLMs, but it also introduces risks, as these plugins developed by various third parties cannot be easily trusted. This paper proposes a new attacking framework to examine security and safety vulnerabilities within LLM platforms that incorporate third-party services. Applying our framework specifically to widely used LLMs, we identify real-world malicious attacks across various domains on third-party APIs that can imperceptibly modify LLM outputs. The paper discusses the unique challenges posed by third-party API integration and offers strategic possibilities to improve the security and safety of LLM ecosystems moving forward. Our code is released at https://github.com/vk0812/Third-Party-Attacks-on-LLMs.



## **17. Talk Too Much: Poisoning Large Language Models under Token Limit**

cs.CL

**SubmitDate**: 2024-04-24    [abs](http://arxiv.org/abs/2404.14795v2) [paper-pdf](http://arxiv.org/pdf/2404.14795v2)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream poisoning attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of the trigger, we present a poisoning attack against LLMs that is triggered by a generation/output condition-token limitation, which is a commonly adopted strategy by users for reducing costs. The poisoned model performs normally for output without token limitation, while becomes harmful for output with limited tokens. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation limitation by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our experiments demonstrate that BrieFool is effective across safety domains and knowledge domains. For instance, with only 20 generated poisoning examples against GPT-3.5-turbo, BrieFool achieves a 100% Attack Success Rate (ASR) and a 9.28/10 average Harmfulness Score (HS) under token limitation conditions while maintaining the benign performance.



## **18. Large Language Models Spot Phishing Emails with Surprising Accuracy: A Comparative Analysis of Performance**

cs.CL

7 pages, 3 figures

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.15485v1) [paper-pdf](http://arxiv.org/pdf/2404.15485v1)

**Authors**: Het Patel, Umair Rehman, Farkhund Iqbal

**Abstract**: Phishing, a prevalent cybercrime tactic for decades, remains a significant threat in today's digital world. By leveraging clever social engineering elements and modern technology, cybercrime targets many individuals, businesses, and organizations to exploit trust and security. These cyber-attackers are often disguised in many trustworthy forms to appear as legitimate sources. By cleverly using psychological elements like urgency, fear, social proof, and other manipulative strategies, phishers can lure individuals into revealing sensitive and personalized information. Building on this pervasive issue within modern technology, this paper aims to analyze the effectiveness of 15 Large Language Models (LLMs) in detecting phishing attempts, specifically focusing on a randomized set of "419 Scam" emails. The objective is to determine which LLMs can accurately detect phishing emails by analyzing a text file containing email metadata based on predefined criteria. The experiment concluded that the following models, ChatGPT 3.5, GPT-3.5-Turbo-Instruct, and ChatGPT, were the most effective in detecting phishing emails.



## **19. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.01318v2) [paper-pdf](http://arxiv.org/pdf/2404.01318v2)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work -- which align with OpenAI's usage policies; (3) a standardized evaluation framework that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community. Over time, we will expand and adapt the benchmark to reflect technical and methodological advances in the research community.



## **20. Versatile Backdoor Attack with Visible, Semantic, Sample-Specific, and Compatible Triggers**

cs.CV

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2306.00816v3) [paper-pdf](http://arxiv.org/pdf/2306.00816v3)

**Authors**: Ruotong Wang, Hongrui Chen, Zihao Zhu, Li Liu, Baoyuan Wu

**Abstract**: Deep neural networks (DNNs) can be manipulated to exhibit specific behaviors when exposed to specific trigger patterns, without affecting their performance on benign samples, dubbed \textit{backdoor attack}. Currently, implementing backdoor attacks in physical scenarios still faces significant challenges. Physical attacks are labor-intensive and time-consuming, and the triggers are selected in a manual and heuristic way. Moreover, expanding digital attacks to physical scenarios faces many challenges due to their sensitivity to visual distortions and the absence of counterparts in the real world. To address these challenges, we define a novel trigger called the \textbf{V}isible, \textbf{S}emantic, \textbf{S}ample-Specific, and \textbf{C}ompatible (VSSC) trigger, to achieve effective, stealthy and robust simultaneously, which can also be effectively deployed in the physical scenario using corresponding objects. To implement the VSSC trigger, we propose an automated pipeline comprising three modules: a trigger selection module that systematically identifies suitable triggers leveraging large language models, a trigger insertion module that employs generative models to seamlessly integrate triggers into images, and a quality assessment module that ensures the natural and successful insertion of triggers through vision-language models. Extensive experimental results and analysis validate the effectiveness, stealthiness, and robustness of the VSSC trigger. It can not only maintain robustness under visual distortions but also demonstrates strong practicality in the physical scenario. We hope that the proposed VSSC trigger and implementation approach could inspire future studies on designing more practical triggers in backdoor attacks.



## **21. Explaining Arguments' Strength: Unveiling the Role of Attacks and Supports (Technical Report)**

cs.AI

This paper has been accepted at IJCAI 2024 (the 33rd International  Joint Conference on Artificial Intelligence)

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14304v1) [paper-pdf](http://arxiv.org/pdf/2404.14304v1)

**Authors**: Xiang Yin, Potyka Nico, Francesca Toni

**Abstract**: Quantitatively explaining the strength of arguments under gradual semantics has recently received increasing attention. Specifically, several works in the literature provide quantitative explanations by computing the attribution scores of arguments. These works disregard the importance of attacks and supports, even though they play an essential role when explaining arguments' strength. In this paper, we propose a novel theory of Relation Attribution Explanations (RAEs), adapting Shapley values from game theory to offer fine-grained insights into the role of attacks and supports in quantitative bipolar argumentation towards obtaining the arguments' strength. We show that RAEs satisfy several desirable properties. We also propose a probabilistic algorithm to approximate RAEs efficiently. Finally, we show the application value of RAEs in fraud detection and large language models case studies.



## **22. Physical Backdoor Attack can Jeopardize Driving with Vision-Large-Language Models**

cs.CR

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.12916v2) [paper-pdf](http://arxiv.org/pdf/2404.12916v2)

**Authors**: Zhenyang Ni, Rui Ye, Yuxi Wei, Zhen Xiang, Yanfeng Wang, Siheng Chen

**Abstract**: Vision-Large-Language-models(VLMs) have great application prospects in autonomous driving. Despite the ability of VLMs to comprehend and make decisions in complex scenarios, their integration into safety-critical autonomous driving systems poses serious security risks. In this paper, we propose BadVLMDriver, the first backdoor attack against VLMs for autonomous driving that can be launched in practice using physical objects. Unlike existing backdoor attacks against VLMs that rely on digital modifications, BadVLMDriver uses common physical items, such as a red balloon, to induce unsafe actions like sudden acceleration, highlighting a significant real-world threat to autonomous vehicle safety. To execute BadVLMDriver, we develop an automated pipeline utilizing natural language instructions to generate backdoor training samples with embedded malicious behaviors. This approach allows for flexible trigger and behavior selection, enhancing the stealth and practicality of the attack in diverse scenarios. We conduct extensive experiments to evaluate BadVLMDriver for two representative VLMs, five different trigger objects, and two types of malicious backdoor behaviors. BadVLMDriver achieves a 92% attack success rate in inducing a sudden acceleration when coming across a pedestrian holding a red balloon. Thus, BadVLMDriver not only demonstrates a critical security risk but also emphasizes the urgent need for developing robust defense mechanisms to protect against such vulnerabilities in autonomous driving technologies.



## **23. Protecting Your LLMs with Information Bottleneck**

cs.CL

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13968v1) [paper-pdf](http://arxiv.org/pdf/2404.13968v1)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.



## **24. Typos that Broke the RAG's Back: Genetic Attack on RAG Pipeline by Simulating Documents in the Wild via Low-level Perturbations**

cs.CL

Under Review

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13948v1) [paper-pdf](http://arxiv.org/pdf/2404.13948v1)

**Authors**: Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho Hwang, Jong C. Park

**Abstract**: The robustness of recent Large Language Models (LLMs) has become increasingly crucial as their applicability expands across various domains and real-world applications. Retrieval-Augmented Generation (RAG) is a promising solution for addressing the limitations of LLMs, yet existing studies on the robustness of RAG often overlook the interconnected relationships between RAG components or the potential threats prevalent in real-world databases, such as minor textual errors. In this work, we investigate two underexplored aspects when assessing the robustness of RAG: 1) vulnerability to noisy documents through low-level perturbations and 2) a holistic evaluation of RAG robustness. Furthermore, we introduce a novel attack method, the Genetic Attack on RAG (\textit{GARAG}), which targets these aspects. Specifically, GARAG is designed to reveal vulnerabilities within each component and test the overall system functionality against noisy documents. We validate RAG robustness by applying our \textit{GARAG} to standard QA datasets, incorporating diverse retrievers and LLMs. The experimental results show that GARAG consistently achieves high attack success rates. Also, it significantly devastates the performance of each component and their synergy, highlighting the substantial risk that minor textual inaccuracies pose in disrupting RAG systems in the real world.



## **25. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

cs.CL

Competition Report

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14461v1) [paper-pdf](http://arxiv.org/pdf/2404.14461v1)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.



## **26. Bot or Human? Detecting ChatGPT Imposters with A Single Question**

cs.CL

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2305.06424v3) [paper-pdf](http://arxiv.org/pdf/2305.06424v3)

**Authors**: Hong Wang, Xuan Luo, Weizhi Wang, Xifeng Yan

**Abstract**: Large language models like GPT-4 have recently demonstrated impressive capabilities in natural language understanding and generation, enabling various applications including translation, essay writing, and chit-chatting. However, there is a concern that they can be misused for malicious purposes, such as fraud or denial-of-service attacks. Therefore, it is crucial to develop methods for detecting whether the party involved in a conversation is a bot or a human. In this paper, we propose a framework named FLAIR, Finding Large Language Model Authenticity via a Single Inquiry and Response, to detect conversational bots in an online manner. Specifically, we target a single question scenario that can effectively differentiate human users from bots. The questions are divided into two categories: those that are easy for humans but difficult for bots (e.g., counting, substitution, and ASCII art reasoning), and those that are easy for bots but difficult for humans (e.g., memorization and computation). Our approach shows different strengths of these questions in their effectiveness, providing a new way for online service providers to protect themselves against nefarious activities and ensure that they are serving real users. We open-sourced our code and dataset on https://github.com/hongwang600/FLAIR and welcome contributions from the community.



## **27. AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs**

cs.CR

32 pages, 9 figures, 7 tables

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.16873v1) [paper-pdf](http://arxiv.org/pdf/2404.16873v1)

**Authors**: Anselm Paulus, Arman Zharmagambetov, Chuan Guo, Brandon Amos, Yuandong Tian

**Abstract**: While recently Large Language Models (LLMs) have achieved remarkable successes, they are vulnerable to certain jailbreaking attacks that lead to generation of inappropriate or harmful content. Manual red-teaming requires finding adversarial prompts that cause such jailbreaking, e.g. by appending a suffix to a given instruction, which is inefficient and time-consuming. On the other hand, automatic adversarial prompt generation often leads to semantically meaningless attacks that can easily be detected by perplexity-based filters, may require gradient information from the TargetLLM, or do not scale well due to time-consuming discrete optimization processes over the token space. In this paper, we present a novel method that uses another LLM, called the AdvPrompter, to generate human-readable adversarial prompts in seconds, $\sim800\times$ faster than existing optimization-based approaches. We train the AdvPrompter using a novel algorithm that does not require access to the gradients of the TargetLLM. This process alternates between two steps: (1) generating high-quality target adversarial suffixes by optimizing the AdvPrompter predictions, and (2) low-rank fine-tuning of the AdvPrompter with the generated adversarial suffixes. The trained AdvPrompter generates suffixes that veil the input instruction without changing its meaning, such that the TargetLLM is lured to give a harmful response. Experimental results on popular open source TargetLLMs show state-of-the-art results on the AdvBench dataset, that also transfer to closed-source black-box LLM APIs. Further, we demonstrate that by fine-tuning on a synthetic dataset generated by AdvPrompter, LLMs can be made more robust against jailbreaking attacks while maintaining performance, i.e. high MMLU scores.



## **28. Trojan Detection in Large Language Models: Insights from The Trojan Detection Challenge**

cs.CL

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.13660v1) [paper-pdf](http://arxiv.org/pdf/2404.13660v1)

**Authors**: Narek Maloyan, Ekansh Verma, Bulat Nutfullin, Bislan Ashinov

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various domains, but their vulnerability to trojan or backdoor attacks poses significant security risks. This paper explores the challenges and insights gained from the Trojan Detection Competition 2023 (TDC2023), which focused on identifying and evaluating trojan attacks on LLMs. We investigate the difficulty of distinguishing between intended and unintended triggers, as well as the feasibility of reverse engineering trojans in real-world scenarios. Our comparative analysis of various trojan detection methods reveals that achieving high Recall scores is significantly more challenging than obtaining high Reverse-Engineering Attack Success Rate (REASR) scores. The top-performing methods in the competition achieved Recall scores around 0.16, comparable to a simple baseline of randomly sampling sentences from a distribution similar to the given training prefixes. This finding raises questions about the detectability and recoverability of trojans inserted into the model, given only the harmful targets. Despite the inability to fully solve the problem, the competition has led to interesting observations about the viability of trojan detection and improved techniques for optimizing LLM input prompts. The phenomenon of unintended triggers and the difficulty in distinguishing them from intended triggers highlights the need for further research into the robustness and interpretability of LLMs. The TDC2023 has provided valuable insights into the challenges and opportunities associated with trojan detection in LLMs, laying the groundwork for future research in this area to ensure their safety and reliability in real-world applications.



## **29. Large Language Models for Blockchain Security: A Systematic Literature Review**

cs.CR

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2403.14280v3) [paper-pdf](http://arxiv.org/pdf/2403.14280v3)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.



## **30. Intrusion Detection at Scale with the Assistance of a Command-line Language Model**

cs.CR

Accepted by IEEE/IFIP International Conference on Dependable Systems  and Networks (DSN), industry track

**SubmitDate**: 2024-04-20    [abs](http://arxiv.org/abs/2404.13402v1) [paper-pdf](http://arxiv.org/pdf/2404.13402v1)

**Authors**: Jiongliang Lin, Yiwen Guo, Hao Chen

**Abstract**: Intrusion detection is a long standing and crucial problem in security. A system capable of detecting intrusions automatically is on great demand in enterprise security solutions. Existing solutions rely heavily on hand-crafted rules designed by security operators, which suffer from high false negative rates and poor generalization ability to new, zero-day attacks at scale. AI and machine learning offer promising solutions to address the issues, by inspecting abnormal user behaviors intelligently and automatically from data. However, existing learning-based intrusion detection systems in the literature are mostly designed for small data, and they lack the ability to leverage the power of big data in cloud environments. In this paper, we target at this problem and introduce an intrusion detection system which incorporates large-scale pre-training, so as to train a large language model based on tens of millions of command lines for AI-based intrusion detection. Experiments performed on 30 million training samples and 10 million test samples verify the effectiveness of our solution.



## **31. ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs**

cs.CL

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2402.11753v3) [paper-pdf](http://arxiv.org/pdf/2402.11753v3)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Zhen Xiang, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran

**Abstract**: Safety is critical to the usage of large language models (LLMs). Multiple techniques such as data filtering and supervised fine-tuning have been developed to strengthen LLM safety. However, currently known techniques presume that corpora used for safety alignment of LLMs are solely interpreted by semantics. This assumption, however, does not hold in real-world applications, which leads to severe vulnerabilities in LLMs. For example, users of forums often use ASCII art, a form of text-based art, to convey image information. In this paper, we propose a novel ASCII art-based jailbreak attack and introduce a comprehensive benchmark Vision-in-Text Challenge (ViTC) to evaluate the capabilities of LLMs in recognizing prompts that cannot be solely interpreted by semantics. We show that five SOTA LLMs (GPT-3.5, GPT-4, Gemini, Claude, and Llama2) struggle to recognize prompts provided in the form of ASCII art. Based on this observation, we develop the jailbreak attack ArtPrompt, which leverages the poor performance of LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs. ArtPrompt only requires black-box access to the victim LLMs, making it a practical attack. We evaluate ArtPrompt on five SOTA LLMs, and show that ArtPrompt can effectively and efficiently induce undesired behaviors from all five LLMs. Our code is available at https://github.com/uw-nsl/ArtPrompt.



## **32. CyberSecEval 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models**

cs.CR

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.13161v1) [paper-pdf](http://arxiv.org/pdf/2404.13161v1)

**Authors**: Manish Bhatt, Sahana Chennabasappa, Yue Li, Cyrus Nikolaidis, Daniel Song, Shengye Wan, Faizan Ahmad, Cornelius Aschermann, Yaohui Chen, Dhaval Kapil, David Molnar, Spencer Whitman, Joshua Saxe

**Abstract**: Large language models (LLMs) introduce new security risks, but there are few comprehensive evaluation suites to measure and reduce these risks. We present BenchmarkName, a novel benchmark to quantify LLM security risks and capabilities. We introduce two new areas for testing: prompt injection and code interpreter abuse. We evaluated multiple state-of-the-art (SOTA) LLMs, including GPT-4, Mistral, Meta Llama 3 70B-Instruct, and Code Llama. Our results show that conditioning away risk of attack remains an unsolved problem; for example, all tested models showed between 26% and 41% successful prompt injection tests. We further introduce the safety-utility tradeoff: conditioning an LLM to reject unsafe prompts can cause the LLM to falsely reject answering benign prompts, which lowers utility. We propose quantifying this tradeoff using False Refusal Rate (FRR). As an illustration, we introduce a novel test set to quantify FRR for cyberattack helpfulness risk. We find many LLMs able to successfully comply with "borderline" benign requests while still rejecting most unsafe requests. Finally, we quantify the utility of LLMs for automating a core cybersecurity task, that of exploiting software vulnerabilities. This is important because the offensive capabilities of LLMs are of intense interest; we quantify this by creating novel test sets for four representative problems. We find that models with coding capabilities perform better than those without, but that further work is needed for LLMs to become proficient at exploit generation. Our code is open source and can be used to evaluate other LLMs.



## **33. Heterogeneous Federated Learning with Splited Language Model**

cs.CV

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2403.16050v2) [paper-pdf](http://arxiv.org/pdf/2403.16050v2)

**Authors**: Yifan Shi, Yuhui Zhang, Ziyue Huang, Xiaofeng Yang, Li Shen, Wei Chen, Xueqian Wang

**Abstract**: Federated Split Learning (FSL) is a promising distributed learning paradigm in practice, which gathers the strengths of both Federated Learning (FL) and Split Learning (SL) paradigms, to ensure model privacy while diminishing the resource overhead of each client, especially on large transformer models in a resource-constrained environment, e.g., Internet of Things (IoT). However, almost all works merely investigate the performance with simple neural network models in FSL. Despite the minor efforts focusing on incorporating Vision Transformers (ViT) as model architectures, they train ViT from scratch, thereby leading to enormous training overhead in each device with limited resources. Therefore, in this paper, we harness Pre-trained Image Transformers (PITs) as the initial model, coined FedV, to accelerate the training process and improve model robustness. Furthermore, we propose FedVZ to hinder the gradient inversion attack, especially having the capability compatible with black-box scenarios, where the gradient information is unavailable. Concretely, FedVZ approximates the server gradient by utilizing a zeroth-order (ZO) optimization, which replaces the backward propagation with just one forward process. Empirically, we are the first to provide a systematic evaluation of FSL methods with PITs in real-world datasets, different partial device participations, and heterogeneous data splits. Our experiments verify the effectiveness of our algorithms.



## **34. A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions**

cs.CL

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2310.14724v3) [paper-pdf](http://arxiv.org/pdf/2310.14724v3)

**Authors**: Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F. Wong, Lidia S. Chao

**Abstract**: The powerful ability to understand, follow, and generate complex language emerging from large language models (LLMs) makes LLM-generated text flood many areas of our daily lives at an incredible speed and is widely accepted by humans. As LLMs continue to expand, there is an imperative need to develop detectors that can detect LLM-generated text. This is crucial to mitigate potential misuse of LLMs and safeguard realms like artistic expression and social networks from harmful influence of LLM-generated content. The LLM-generated text detection aims to discern if a piece of text was produced by an LLM, which is essentially a binary classification task. The detector techniques have witnessed notable advancements recently, propelled by innovations in watermarking techniques, statistics-based detectors, neural-base detectors, and human-assisted methods. In this survey, we collate recent research breakthroughs in this area and underscore the pressing need to bolster detector research. We also delve into prevalent datasets, elucidating their limitations and developmental requirements. Furthermore, we analyze various LLM-generated text detection paradigms, shedding light on challenges like out-of-distribution problems, potential attacks, real-world data issues and the lack of effective evaluation framework. Conclusively, we highlight interesting directions for future research in LLM-generated text detection to advance the implementation of responsible artificial intelligence (AI). Our aim with this survey is to provide a clear and comprehensive introduction for newcomers while also offering seasoned researchers a valuable update in the field of LLM-generated text detection. The useful resources are publicly available at: https://github.com/NLP2CT/LLM-generated-Text-Detection.



## **35. JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.03027v2) [paper-pdf](http://arxiv.org/pdf/2404.03027v2)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.



## **36. Advancing the Robustness of Large Language Models through Self-Denoised Smoothing**

cs.CL

Accepted by NAACL 2024. Jiabao, Bairu, Zhen, Guanhua contributed  equally. This is an updated version of the paper: arXiv:2307.07171

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12274v1) [paper-pdf](http://arxiv.org/pdf/2404.12274v1)

**Authors**: Jiabao Ji, Bairu Hou, Zhen Zhang, Guanhua Zhang, Wenqi Fan, Qing Li, Yang Zhang, Gaowen Liu, Sijia Liu, Shiyu Chang

**Abstract**: Although large language models (LLMs) have achieved significant success, their vulnerability to adversarial perturbations, including recent jailbreak attacks, has raised considerable concerns. However, the increasing size of these models and their limited access make improving their robustness a challenging task. Among various defense strategies, randomized smoothing has shown great potential for LLMs, as it does not require full access to the model's parameters or fine-tuning via adversarial training. However, randomized smoothing involves adding noise to the input before model prediction, and the final model's robustness largely depends on the model's performance on these noise corrupted data. Its effectiveness is often limited by the model's sub-optimal performance on noisy data. To address this issue, we propose to leverage the multitasking nature of LLMs to first denoise the noisy inputs and then to make predictions based on these denoised versions. We call this procedure self-denoised smoothing. Unlike previous denoised smoothing techniques in computer vision, which require training a separate model to enhance the robustness of LLMs, our method offers significantly better efficiency and flexibility. Our experimental results indicate that our method surpasses existing methods in both empirical and certified robustness in defending against adversarial attacks for both downstream tasks and human alignments (i.e., jailbreak attacks). Our code is publicly available at https://github.com/UCSB-NLP-Chang/SelfDenoise



## **37. Concept Induction: Analyzing Unstructured Text with High-Level Concepts Using LLooM**

cs.HC

To appear at CHI 2024

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12259v1) [paper-pdf](http://arxiv.org/pdf/2404.12259v1)

**Authors**: Michelle S. Lam, Janice Teoh, James Landay, Jeffrey Heer, Michael S. Bernstein

**Abstract**: Data analysts have long sought to turn unstructured text data into meaningful concepts. Though common, topic modeling and clustering focus on lower-level keywords and require significant interpretative work. We introduce concept induction, a computational process that instead produces high-level concepts, defined by explicit inclusion criteria, from unstructured text. For a dataset of toxic online comments, where a state-of-the-art BERTopic model outputs "women, power, female," concept induction produces high-level concepts such as "Criticism of traditional gender roles" and "Dismissal of women's concerns." We present LLooM, a concept induction algorithm that leverages large language models to iteratively synthesize sampled text and propose human-interpretable concepts of increasing generality. We then instantiate LLooM in a mixed-initiative text analysis tool, enabling analysts to shift their attention from interpreting topics to engaging in theory-driven analysis. Through technical evaluations and four analysis scenarios ranging from literature review to content moderation, we find that LLooM's concepts improve upon the prior art of topic models in terms of quality and data coverage. In expert case studies, LLooM helped researchers to uncover new insights even from familiar datasets, for example by suggesting a previously unnoticed concept of attacks on out-party stances in a political social media dataset.



## **38. Efficiently Adversarial Examples Generation for Visual-Language Models under Targeted Transfer Scenarios using Diffusion Models**

cs.CV

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.10335v2) [paper-pdf](http://arxiv.org/pdf/2404.10335v2)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Qing Guo

**Abstract**: Targeted transfer-based attacks involving adversarial examples pose a significant threat to large visual-language models (VLMs). However, the state-of-the-art (SOTA) transfer-based attacks incur high costs due to excessive iteration counts. Furthermore, the generated adversarial examples exhibit pronounced adversarial noise and demonstrate limited efficacy in evading defense methods such as DiffPure. To address these issues, inspired by score matching, we introduce AdvDiffVLM, which utilizes diffusion models to generate natural, unrestricted adversarial examples. Specifically, AdvDiffVLM employs Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring the adversarial examples produced contain natural adversarial semantics and thus possess enhanced transferability. Simultaneously, to enhance the quality of adversarial examples further, we employ the GradCAM-guided Mask method to disperse adversarial semantics throughout the image, rather than concentrating them in a specific area. Experimental results demonstrate that our method achieves a speedup ranging from 10X to 30X compared to existing transfer-based attack methods, while maintaining superior quality of adversarial examples. Additionally, the generated adversarial examples possess strong transferability and exhibit increased robustness against adversarial defense methods. Notably, AdvDiffVLM can successfully attack commercial VLMs, including GPT-4V, in a black-box manner.



## **39. Uncovering Safety Risks in Open-source LLMs through Concept Activation Vector**

cs.CL

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12038v1) [paper-pdf](http://arxiv.org/pdf/2404.12038v1)

**Authors**: Zhihao Xu, Ruixuan Huang, Xiting Wang, Fangzhao Wu, Jing Yao, Xing Xie

**Abstract**: Current open-source large language models (LLMs) are often undergone careful safety alignment before public release. Some attack methods have also been proposed that help check for safety vulnerabilities in LLMs to ensure alignment robustness. However, many of these methods have moderate attack success rates. Even when successful, the harmfulness of their outputs cannot be guaranteed, leading to suspicions that these methods have not accurately identified the safety vulnerabilities of LLMs. In this paper, we introduce a LLM attack method utilizing concept-based model explanation, where we extract safety concept activation vectors (SCAVs) from LLMs' activation space, enabling efficient attacks on well-aligned LLMs like LLaMA-2, achieving near 100% attack success rate as if LLMs are completely unaligned. This suggests that LLMs, even after thorough safety alignment, could still pose potential risks to society upon public release. To evaluate the harmfulness of outputs resulting with various attack methods, we propose a comprehensive evaluation method that reduces the potential inaccuracies of existing evaluations, and further validate that our method causes more harmful content. Additionally, we discover that the SCAVs show some transferability across different open-source LLMs.



## **40. Sampling-based Pseudo-Likelihood for Membership Inference Attacks**

cs.CL

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2404.11262v1) [paper-pdf](http://arxiv.org/pdf/2404.11262v1)

**Authors**: Masahiro Kaneko, Youmi Ma, Yuki Wata, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) are trained on large-scale web data, which makes it difficult to grasp the contribution of each text. This poses the risk of leaking inappropriate data such as benchmarks, personal information, and copyrighted texts in the training data. Membership Inference Attacks (MIA), which determine whether a given text is included in the model's training data, have been attracting attention. Previous studies of MIAs revealed that likelihood-based classification is effective for detecting leaks in LLMs. However, the existing methods cannot be applied to some proprietary models like ChatGPT or Claude 3 because the likelihood is unavailable to the user. In this study, we propose a Sampling-based Pseudo-Likelihood (\textbf{SPL}) method for MIA (\textbf{SaMIA}) that calculates SPL using only the text generated by an LLM to detect leaks. The SaMIA treats the target text as the reference text and multiple outputs from the LLM as text samples, calculates the degree of $n$-gram match as SPL, and determines the membership of the text in the training data. Even without likelihoods, SaMIA performed on par with existing likelihood-based methods.



## **41. Humans or LLMs as the Judge? A Study on Judgement Biases**

cs.CL

22 pages

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2402.10669v3) [paper-pdf](http://arxiv.org/pdf/2402.10669v3)

**Authors**: Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, Benyou Wang

**Abstract**: Adopting human and large language models (LLM) as judges (\textit{a.k.a} human- and LLM-as-a-judge) for evaluating the performance of LLMs has recently gained attention. Nonetheless, this approach concurrently introduces potential biases from human and LLM judges, questioning the reliability of the evaluation results. In this paper, we propose a novel framework that is free from referencing groundtruth annotations for investigating Fallacy Oversight Bias, Authority Bias and Beauty Bias on LLM and human judges. We curate a dataset referring to the revised Bloom's Taxonomy and conduct thousands of human and LLM evaluations. Results show that human and LLM judges are vulnerable to perturbations to various degrees, and that even the cutting-edge judges possess considerable biases. We further exploit their weakness and conduct attacks on LLM judges. We hope that our work can notify the community of the vulnerability of human- and LLM-as-a-judge against perturbations, as well as the urgency of developing robust evaluation systems.



## **42. TransLinkGuard: Safeguarding Transformer Models Against Model Stealing in Edge Deployment**

cs.CR

arXiv admin note: text overlap with arXiv:2310.07152 by other authors

**SubmitDate**: 2024-04-17    [abs](http://arxiv.org/abs/2404.11121v1) [paper-pdf](http://arxiv.org/pdf/2404.11121v1)

**Authors**: Qinfeng Li, Zhiqiang Shen, Zhenghan Qin, Yangfan Xie, Xuhong Zhang, Tianyu Du, Jianwei Yin

**Abstract**: Proprietary large language models (LLMs) have been widely applied in various scenarios. Additionally, deploying LLMs on edge devices is trending for efficiency and privacy reasons. However, edge deployment of proprietary LLMs introduces new security challenges: edge-deployed models are exposed as white-box accessible to users, enabling adversaries to conduct effective model stealing (MS) attacks. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify four critical protection properties that existing methods fail to simultaneously satisfy: (1) maintaining protection after a model is physically copied; (2) authorizing model access at request level; (3) safeguarding runtime reverse engineering; (4) achieving high security with negligible runtime overhead. To address the above issues, we propose TransLinkGuard, a plug-and-play model protection approach against model stealing on edge devices. The core part of TransLinkGuard is a lightweight authorization module residing in a secure environment, e.g., TEE. The authorization module can freshly authorize each request based on its input. Extensive experiments show that TransLinkGuard achieves the same security protection as the black-box security guarantees with negligible overhead.



## **43. Hidden You Malicious Goal Into Benign Narratives: Jailbreak Large Language Models through Logic Chain Injection**

cs.CR

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2404.04849v2) [paper-pdf](http://arxiv.org/pdf/2404.04849v2)

**Authors**: Zhilong Wang, Yebo Cao, Peng Liu

**Abstract**: Jailbreak attacks on Language Model Models (LLMs) entail crafting prompts aimed at exploiting the models to generate malicious content. Existing jailbreak attacks can successfully deceive the LLMs, however they cannot deceive the human. This paper proposes a new type of jailbreak attacks which can deceive both the LLMs and human (i.e., security analyst). The key insight of our idea is borrowed from the social psychology - that is human are easily deceived if the lie is hidden in truth. Based on this insight, we proposed the logic-chain injection attacks to inject malicious intention into benign truth. Logic-chain injection attack firstly dissembles its malicious target into a chain of benign narrations, and then distribute narrations into a related benign article, with undoubted facts. In this way, newly generate prompt cannot only deceive the LLMs, but also deceive human.



## **44. IsamasRed: A Public Dataset Tracking Reddit Discussions on Israel-Hamas Conflict**

cs.SI

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2401.08202v2) [paper-pdf](http://arxiv.org/pdf/2401.08202v2)

**Authors**: Kai Chen, Zihao He, Keith Burghardt, Jingxin Zhang, Kristina Lerman

**Abstract**: The conflict between Israel and Palestinians significantly escalated after the October 7, 2023 Hamas attack, capturing global attention. To understand the public discourse on this conflict, we present a meticulously compiled dataset-IsamasRed-comprising nearly 400,000 conversations and over 8 million comments from Reddit, spanning from August 2023 to November 2023. We introduce an innovative keyword extraction framework leveraging a large language model to effectively identify pertinent keywords, ensuring a comprehensive data collection. Our initial analysis on the dataset, examining topics, controversy, emotional and moral language trends over time, highlights the emotionally charged and complex nature of the discourse. This dataset aims to enrich the understanding of online discussions, shedding light on the complex interplay between ideology, sentiment, and community engagement in digital spaces.



## **45. Self-playing Adversarial Language Game Enhances LLM Reasoning**

cs.CL

Preprint

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2404.10642v1) [paper-pdf](http://arxiv.org/pdf/2404.10642v1)

**Authors**: Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du

**Abstract**: We explore the self-play training procedure of large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate with respect to a target word only visible to the attacker. The attacker aims to induce the defender to utter the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players should have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by Self-Play in this Adversarial language Game (SPAG). With this goal, we let LLMs act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performance uniformly improves on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLM's reasoning ability. The code is at https://github.com/Linear95/SPAG.



## **46. Topic-based Watermarks for LLM-Generated Text**

cs.CR

11 pages

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2404.02138v2) [paper-pdf](http://arxiv.org/pdf/2404.02138v2)

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday

**Abstract**: Recent advancements of large language models (LLMs) have resulted in indistinguishable text outputs comparable to human-generated text. Watermarking algorithms are potential tools that offer a way to differentiate between LLM- and human-generated text by embedding detectable signatures within LLM-generated output. However, current watermarking schemes lack robustness against known attacks against watermarking algorithms. In addition, they are impractical considering an LLM generates tens of thousands of text outputs per day and the watermarking algorithm needs to memorize each output it generates for the detection to work. In this work, focusing on the limitations of current watermarking schemes, we propose the concept of a "topic-based watermarking algorithm" for LLMs. The proposed algorithm determines how to generate tokens for the watermarked LLM output based on extracted topics of an input prompt or the output of a non-watermarked LLM. Inspired from previous work, we propose using a pair of lists (that are generated based on the specified extracted topic(s)) that specify certain tokens to be included or excluded while generating the watermarked output of the LLM. Using the proposed watermarking algorithm, we show the practicality of a watermark detection algorithm. Furthermore, we discuss a wide range of attacks that can emerge against watermarking algorithms for LLMs and the benefit of the proposed watermarking scheme for the feasibility of modeling a potential attacker considering its benefit vs. loss.



## **47. Provably Robust Multi-bit Watermarking for AI-generated Text via Error Correction Code**

cs.CR

**SubmitDate**: 2024-04-16    [abs](http://arxiv.org/abs/2401.16820v2) [paper-pdf](http://arxiv.org/pdf/2401.16820v2)

**Authors**: Wenjie Qu, Dong Yin, Zixin He, Wei Zou, Tianyang Tao, Jinyuan Jia, Jiaheng Zhang

**Abstract**: Large Language Models (LLMs) have been widely deployed for their remarkable capability to generate texts resembling human language. However, they could be misused by criminals to create deceptive content, such as fake news and phishing emails, which raises ethical concerns. Watermarking is a key technique to mitigate the misuse of LLMs, which embeds a watermark (e.g., a bit string) into a text generated by a LLM. Consequently, this enables the detection of texts generated by a LLM as well as the tracing of generated texts to a specific user. The major limitation of existing watermark techniques is that they cannot accurately or efficiently extract the watermark from a text, especially when the watermark is a long bit string. This key limitation impedes their deployment for real-world applications, e.g., tracing generated texts to a specific user.   This work introduces a novel watermarking method for LLM-generated text grounded in \textbf{error-correction codes} to address this challenge. We provide strong theoretical analysis, demonstrating that under bounded adversarial word/token edits (insertion, deletion, and substitution), our method can correctly extract watermarks, offering a provable robustness guarantee. This breakthrough is also evidenced by our extensive experimental results. The experiments show that our method substantially outperforms existing baselines in both accuracy and robustness on benchmark datasets. For instance, when embedding a bit string of length 12 into a 200-token generated text, our approach attains an impressive match rate of $98.4\%$, surpassing the performance of Yoo et al. (state-of-the-art baseline) at $85.6\%$. When subjected to a copy-paste attack involving the injection of 50 tokens to generated texts with 200 words, our method maintains a substantial match rate of $90.8\%$, while the match rate of Yoo et al. diminishes to below $65\%$.



## **48. FuzzLLM: A Novel and Universal Fuzzing Framework for Proactively Discovering Jailbreak Vulnerabilities in Large Language Models**

cs.CR

Publish by ICASSP 2024 on 3/18/2024; Extended Arxiv version

**SubmitDate**: 2024-04-14    [abs](http://arxiv.org/abs/2309.05274v2) [paper-pdf](http://arxiv.org/pdf/2309.05274v2)

**Authors**: Dongyu Yao, Jianshu Zhang, Ian G. Harris, Marcel Carlsson

**Abstract**: Jailbreak vulnerabilities in Large Language Models (LLMs), which exploit meticulously crafted prompts to elicit content that violates service guidelines, have captured the attention of research communities. While model owners can defend against individual jailbreak prompts through safety training strategies, this relatively passive approach struggles to handle the broader category of similar jailbreaks. To tackle this issue, we introduce FuzzLLM, an automated fuzzing framework designed to proactively test and discover jailbreak vulnerabilities in LLMs. We utilize templates to capture the structural integrity of a prompt and isolate key features of a jailbreak class as constraints. By integrating different base classes into powerful combo attacks and varying the elements of constraints and prohibited questions, FuzzLLM enables efficient testing with reduced manual effort. Extensive experiments demonstrate FuzzLLM's effectiveness and comprehensiveness in vulnerability discovery across various LLMs.



## **49. Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models**

cs.CV

Work in progress

**SubmitDate**: 2024-04-14    [abs](http://arxiv.org/abs/2403.09792v2) [paper-pdf](http://arxiv.org/pdf/2403.09792v2)

**Authors**: Yifan Li, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, Ji-Rong Wen

**Abstract**: In this paper, we study the harmlessness alignment problem of multimodal large language models (MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method named HADES, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images. Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate (ASR) of 90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision. Our code and data will be publicly released.



## **50. Detoxifying Large Language Models via Knowledge Editing**

cs.CL

Ongoing work. Project website:  https://zjunlp.github.io/project/SafeEdit Add and update experimental results  in Tables 1 and 3

**SubmitDate**: 2024-04-13    [abs](http://arxiv.org/abs/2403.14472v3) [paper-pdf](http://arxiv.org/pdf/2403.14472v3)

**Authors**: Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, Huajun Chen

**Abstract**: This paper investigates using knowledge editing techniques to detoxify Large Language Models (LLMs). We construct a benchmark, SafeEdit, which covers nine unsafe categories with various powerful attack prompts and equips comprehensive metrics for systematic evaluation. We conduct experiments with several knowledge editing approaches, indicating that knowledge editing has the potential to efficiently detoxify LLMs with limited impact on general performance. Then, we propose a simple yet effective baseline, dubbed Detoxifying with Intraoperative Neural Monitoring (DINM), to diminish the toxicity of LLMs within a few tuning steps via only one instance. We further provide an in-depth analysis of the internal mechanism for various detoxifying approaches, demonstrating that previous methods like SFT and DPO may merely suppress the activations of toxic parameters, while DINM mitigates the toxicity of the toxic parameters to a certain extent, making permanent adjustments. We hope that these insights could shed light on future work of developing detoxifying approaches and the underlying knowledge mechanisms of LLMs. Code and benchmark are available at https://github.com/zjunlp/EasyEdit.



