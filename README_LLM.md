# Latest Large Language Model Attack Papers
**update at 2024-11-04 11:07:58**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Efficient Adversarial Training in LLMs with Continuous Attacks**

cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.15589v3) [paper-pdf](http://arxiv.org/pdf/2405.15589v3)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on five models from different families (Gemma, Phi3, Mistral, Zephyr, Llama2) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.



## **2. Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level**

cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.16405v2) [paper-pdf](http://arxiv.org/pdf/2405.16405v2)

**Authors**: Runlin Lei, Yuwei Hu, Yuchen Ren, Zhewei Wei

**Abstract**: Graph Neural Networks (GNNs) excel across various applications but remain vulnerable to adversarial attacks, particularly Graph Injection Attacks (GIAs), which inject malicious nodes into the original graph and pose realistic threats. Text-attributed graphs (TAGs), where nodes are associated with textual features, are crucial due to their prevalence in real-world applications and are commonly used to evaluate these vulnerabilities. However, existing research only focuses on embedding-level GIAs, which inject node embeddings rather than actual textual content, limiting their applicability and simplifying detection. In this paper, we pioneer the exploration of GIAs at the text level, presenting three novel attack designs that inject textual content into the graph. Through theoretical and empirical analysis, we demonstrate that text interpretability, a factor previously overlooked at the embedding level, plays a crucial role in attack strength. Among the designs we investigate, the Word-frequency-based Text-level GIA (WTGIA) is particularly notable for its balance between performance and interpretability. Despite the success of WTGIA, we discover that defenders can easily enhance their defenses with customized text embedding methods or large language model (LLM)--based predictors. These insights underscore the necessity for further research into the potential and practical significance of text-level GIAs.



## **3. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

cs.CR

9 pages

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00459v1) [paper-pdf](http://arxiv.org/pdf/2411.00459v1)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.



## **4. Attention Tracker: Detecting Prompt Injection Attacks in LLMs**

cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00348v1) [paper-pdf](http://arxiv.org/pdf/2411.00348v1)

**Authors**: Kuo-Han Hung, Ching-Yun Ko, Ambrish Rawat, I-Hsin Chung, Winston H. Hsu, Pin-Yu Chen

**Abstract**: Large Language Models (LLMs) have revolutionized various domains but remain vulnerable to prompt injection attacks, where malicious inputs manipulate the model into ignoring original instructions and executing designated action. In this paper, we investigate the underlying mechanisms of these attacks by analyzing the attention patterns within LLMs. We introduce the concept of the distraction effect, where specific attention heads, termed important heads, shift focus from the original instruction to the injected instruction. Building on this discovery, we propose Attention Tracker, a training-free detection method that tracks attention patterns on instruction to detect prompt injection attacks without the need for additional LLM inference. Our method generalizes effectively across diverse models, datasets, and attack types, showing an AUROC improvement of up to 10.0% over existing methods, and performs well even on small LLMs. We demonstrate the robustness of our approach through extensive evaluations and provide insights into safeguarding LLM-integrated systems from prompt injection vulnerabilities.



## **5. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

The camera-ready version of JailbreakBench v1.0 (accepted at NeurIPS  2024 Datasets and Benchmarks Track): more attack artifacts, more test-time  defenses, a more accurate jailbreak judge (Llama-3-70B with a custom prompt),  a larger dataset of human preferences for selecting a jailbreak judge (300  examples), an over-refusal evaluation dataset, a semantic refusal judge based  on Llama-3-8B

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2404.01318v5) [paper-pdf](http://arxiv.org/pdf/2404.01318v5)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.



## **6. Scaling Up Membership Inference: When and How Attacks Succeed on Large Language Models**

cs.CL

Our code is available at https://github.com/parameterlab/mia-scaling

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2411.00154v1) [paper-pdf](http://arxiv.org/pdf/2411.00154v1)

**Authors**: Haritz Puerto, Martin Gubri, Sangdoo Yun, Seong Joon Oh

**Abstract**: Membership inference attacks (MIA) attempt to verify the membership of a given data sample in the training set for a model. MIA has become relevant in recent years, following the rapid development of large language models (LLM). Many are concerned about the usage of copyrighted materials for training them and call for methods for detecting such usage. However, recent research has largely concluded that current MIA methods do not work on LLMs. Even when they seem to work, it is usually because of the ill-designed experimental setup where other shortcut features enable "cheating." In this work, we argue that MIA still works on LLMs, but only when multiple documents are presented for testing. We construct new benchmarks that measure the MIA performances at a continuous scale of data samples, from sentences (n-grams) to a collection of documents (multiple chunks of tokens). To validate the efficacy of current MIA approaches at greater scales, we adapt a recent work on Dataset Inference (DI) for the task of binary membership detection that aggregates paragraph-level MIA features to enable MIA at document and collection of documents level. This baseline achieves the first successful MIA on pre-trained and fine-tuned LLMs.



## **7. Tree of Attacks: Jailbreaking Black-Box LLMs Automatically**

cs.LG

Accepted for presentation at NeurIPS 2024. Code:  https://github.com/RICommunity/TAP

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2312.02119v3) [paper-pdf](http://arxiv.org/pdf/2312.02119v3)

**Authors**: Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi

**Abstract**: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an attacker LLM to iteratively refine candidate (attack) prompts until one of the refined prompts jailbreaks the target. In addition, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks, reducing the number of queries sent to the target LLM. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4-Turbo and GPT4o) for more than 80% of the prompts. This significantly improves upon the previous state-of-the-art black-box methods for generating jailbreaks while using a smaller number of queries than them. Furthermore, TAP is also capable of jailbreaking LLMs protected by state-of-the-art guardrails, e.g., LlamaGuard.



## **8. Desert Camels and Oil Sheikhs: Arab-Centric Red Teaming of Frontier LLMs**

cs.CL

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.24049v1) [paper-pdf](http://arxiv.org/pdf/2410.24049v1)

**Authors**: Muhammed Saeed, Elgizouli Mohamed, Mukhtar Mohamed, Shaina Raza, Shady Shehata, Muhammad Abdul-Mageed

**Abstract**: Large language models (LLMs) are widely used but raise ethical concerns due to embedded social biases. This study examines LLM biases against Arabs versus Westerners across eight domains, including women's rights, terrorism, and anti-Semitism and assesses model resistance to perpetuating these biases. To this end, we create two datasets: one to evaluate LLM bias toward Arabs versus Westerners and another to test model safety against prompts that exaggerate negative traits ("jailbreaks"). We evaluate six LLMs -- GPT-4, GPT-4o, LlaMA 3.1 (8B & 405B), Mistral 7B, and Claude 3.5 Sonnet. We find 79% of cases displaying negative biases toward Arabs, with LlaMA 3.1-405B being the most biased. Our jailbreak tests reveal GPT-4o as the most vulnerable, despite being an optimized version, followed by LlaMA 3.1-8B and Mistral 7B. All LLMs except Claude exhibit attack success rates above 87% in three categories. We also find Claude 3.5 Sonnet the safest, but it still displays biases in seven of eight categories. Despite being an optimized version of GPT4, We find GPT-4o to be more prone to biases and jailbreaks, suggesting optimization flaws. Our findings underscore the pressing need for more robust bias mitigation strategies and strengthened security measures in LLMs.



## **9. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

cs.LG

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2402.06255v4) [paper-pdf](http://arxiv.org/pdf/2402.06255v4)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreaking attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly focusing on model fine-tuning or heuristical defense designs. However, how to achieve intrinsic robustness through prompt optimization remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both grey-box and black-box attacks, reducing the success rate of advanced attacks to nearly 0%, while maintaining the model's utility on the benign task and incurring only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/PKU-ML/PAT.



## **10. Audio Is the Achilles' Heel: Red Teaming Audio Large Multimodal Models**

cs.CL

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23861v1) [paper-pdf](http://arxiv.org/pdf/2410.23861v1)

**Authors**: Hao Yang, Lizhen Qu, Ehsan Shareghi, Gholamreza Haffari

**Abstract**: Large Multimodal Models (LMMs) have demonstrated the ability to interact with humans under real-world conditions by combining Large Language Models (LLMs) and modality encoders to align multimodal information (visual and auditory) with text. However, such models raise new safety challenges of whether models that are safety-aligned on text also exhibit consistent safeguards for multimodal inputs. Despite recent safety-alignment research on vision LMMs, the safety of audio LMMs remains under-explored. In this work, we comprehensively red team the safety of five advanced audio LMMs under three settings: (i) harmful questions in both audio and text formats, (ii) harmful questions in text format accompanied by distracting non-speech audio, and (iii) speech-specific jailbreaks. Our results under these settings demonstrate that open-source audio LMMs suffer an average attack success rate of 69.14% on harmful audio questions, and exhibit safety vulnerabilities when distracted with non-speech audio noise. Our speech-specific jailbreaks on Gemini-1.5-Pro achieve an attack success rate of 70.67% on the harmful query benchmark. We provide insights on what could cause these reported safety-misalignments. Warning: this paper contains offensive examples.



## **11. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

cs.CL

Accepted to NeurIPS 2024 Dataset & Benchmarking Track

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23746v1) [paper-pdf](http://arxiv.org/pdf/2410.23746v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating advanced prompt usages, human revisions like word substitutions, and writing errors. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.



## **12. Adversarial Attacks of Vision Tasks in the Past 10 Years: A Survey**

cs.CV

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23687v1) [paper-pdf](http://arxiv.org/pdf/2410.23687v1)

**Authors**: Chiyu Zhang, Xiaogang Xu, Jiafei Wu, Zhe Liu, Lu Zhou

**Abstract**: Adversarial attacks, which manipulate input data to undermine model availability and integrity, pose significant security threats during machine learning inference. With the advent of Large Vision-Language Models (LVLMs), new attack vectors, such as cognitive bias, prompt injection, and jailbreak techniques, have emerged. Understanding these attacks is crucial for developing more robust systems and demystifying the inner workings of neural networks. However, existing reviews often focus on attack classifications and lack comprehensive, in-depth analysis. The research community currently needs: 1) unified insights into adversariality, transferability, and generalization; 2) detailed evaluations of existing methods; 3) motivation-driven attack categorizations; and 4) an integrated perspective on both traditional and LVLM attacks. This article addresses these gaps by offering a thorough summary of traditional and LVLM adversarial attacks, emphasizing their connections and distinctions, and providing actionable insights for future research.



## **13. Pseudo-Conversation Injection for LLM Goal Hijacking**

cs.CL

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23678v1) [paper-pdf](http://arxiv.org/pdf/2410.23678v1)

**Authors**: Zheng Chen, Buhui Yao

**Abstract**: Goal hijacking is a type of adversarial attack on Large Language Models (LLMs) where the objective is to manipulate the model into producing a specific, predetermined output, regardless of the user's original input. In goal hijacking, an attacker typically appends a carefully crafted malicious suffix to the user's prompt, which coerces the model into ignoring the user's original input and generating the target response. In this paper, we introduce a novel goal hijacking attack method called Pseudo-Conversation Injection, which leverages the weaknesses of LLMs in role identification within conversation contexts. Specifically, we construct the suffix by fabricating responses from the LLM to the user's initial prompt, followed by a prompt for a malicious new task. This leads the model to perceive the initial prompt and fabricated response as a completed conversation, thereby executing the new, falsified prompt. Following this approach, we propose three Pseudo-Conversation construction strategies: Targeted Pseudo-Conversation, Universal Pseudo-Conversation, and Robust Pseudo-Conversation. These strategies are designed to achieve effective goal hijacking across various scenarios. Our experiments, conducted on two mainstream LLM platforms including ChatGPT and Qwen, demonstrate that our proposed method significantly outperforms existing approaches in terms of attack effectiveness.



## **14. Adversarial Attacks on Code Models with Discriminative Graph Patterns**

cs.SE

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2308.11161v2) [paper-pdf](http://arxiv.org/pdf/2308.11161v2)

**Authors**: Thanh-Dat Nguyen, Yang Zhou, Xuan Bach D. Le, Patanamon Thongtanunam, David Lo

**Abstract**: Pre-trained language models of code are now widely used in various software engineering tasks such as code generation, code completion, vulnerability detection, etc. This, in turn, poses security and reliability risks to these models. One of the important threats is \textit{adversarial attacks}, which can lead to erroneous predictions and largely affect model performance on downstream tasks. Current adversarial attacks on code models usually adopt fixed sets of program transformations, such as variable renaming and dead code insertion, leading to limited attack effectiveness. To address the aforementioned challenges, we propose a novel adversarial attack framework, GraphCodeAttack, to better evaluate the robustness of code models. Given a target code model, GraphCodeAttack automatically mines important code patterns, which can influence the model's decisions, to perturb the structure of input code to the model. To do so, GraphCodeAttack uses a set of input source codes to probe the model's outputs and identifies the \textit{discriminative} ASTs patterns that can influence the model decisions. GraphCodeAttack then selects appropriate AST patterns, concretizes the selected patterns as attacks, and inserts them as dead code into the model's input program. To effectively synthesize attacks from AST patterns, GraphCodeAttack uses a separate pre-trained code model to fill in the ASTs with concrete code snippets. We evaluate the robustness of two popular code models (e.g., CodeBERT and GraphCodeBERT) against our proposed approach on three tasks: Authorship Attribution, Vulnerability Prediction, and Clone Detection. The experimental results suggest that our proposed approach significantly outperforms state-of-the-art approaches in attacking code models such as CARROT and ALERT.



## **15. Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning**

cs.LG

Proceedings of the 7th BlackboxNLP Workshop: Analyzing and  Interpreting Neural Networks for NLP

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2401.10862v3) [paper-pdf](http://arxiv.org/pdf/2401.10862v3)

**Authors**: Adib Hasan, Ileana Rugina, Alex Wang

**Abstract**: This paper investigates the impact of model compression on the way Large Language Models (LLMs) process prompts, particularly concerning jailbreak resistance. We show that moderate WANDA pruning can enhance resistance to jailbreaking attacks without fine-tuning, while maintaining performance on standard benchmarks. To systematically evaluate this safety enhancement, we introduce a dataset of 225 harmful tasks across five categories. Our analysis of LLaMA-2 Chat, Vicuna 1.3, and Mistral Instruct v0.2 reveals that pruning benefits correlate with initial model safety levels. We interpret these results by examining changes in attention patterns and perplexity shifts, demonstrating that pruned models exhibit sharper attention and increased sensitivity to artificial jailbreak constructs. We extend our evaluation to the AdvBench harmful behavior tasks and the GCG attack method. We find that LLaMA-2 is much safer on AdvBench prompts than on our dataset when evaluated with manual jailbreak attempts, and that pruning is effective against both automated attacks and manual jailbreaking on Advbench.



## **16. HuRef: HUman-REadable Fingerprint for Large Language Models**

cs.CL

NeurIPS2024

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2312.04828v3) [paper-pdf](http://arxiv.org/pdf/2312.04828v3)

**Authors**: Boyi Zeng, Lizheng Wang, Yuncong Hu, Yi Xu, Chenghu Zhou, Xinbing Wang, Yu Yu, Zhouhan Lin

**Abstract**: Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations. In this study, we introduce HuRef, a human-readable fingerprint for LLMs that uniquely identifies the base model without interfering with training or exposing model parameters to the public. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, with negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning, and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension permutation or matrix rotation, which significantly change it without affecting performance. To address this, leveraging the Transformer structure, we systematically analyze potential attacks and define three invariant terms that identify an LLM's base model. Due to the potential risk of information leakage, we cannot publish invariant terms directly. Instead, we map them to a Gaussian vector using an encoder, then convert it into a natural image using StyleGAN2, and finally publish the image. In our black-box setting, all fingerprinting steps are internally conducted by the LLMs owners. To ensure the published fingerprints are honestly generated, we introduced Zero-Knowledge Proof (ZKP). Experimental results across various LLMs demonstrate the effectiveness of our method. The code is available at https://github.com/LUMIA-Group/HuRef.



## **17. Transferable Ensemble Black-box Jailbreak Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23558v1) [paper-pdf](http://arxiv.org/pdf/2410.23558v1)

**Authors**: Yiqi Yang, Hongye Fu

**Abstract**: In this report, we propose a novel black-box jailbreak attacking framework that incorporates various LLM-as-Attacker methods to deliver transferable and powerful jailbreak attacks. Our method is designed based on three key observations from existing jailbreaking studies and practices. First, we consider an ensemble approach should be more effective in exposing the vulnerabilities of an aligned LLM compared to individual attacks. Second, different malicious instructions inherently vary in their jailbreaking difficulty, necessitating differentiated treatment to ensure more efficient attacks. Finally, the semantic coherence of a malicious instruction is crucial for triggering the defenses of an aligned LLM; therefore, it must be carefully disrupted to manipulate its embedding representation, thereby increasing the jailbreak success rate. We validated our approach by participating in the Competition for LLM and Agent Safety 2024, where our team achieved top performance in the Jailbreaking Attack Track.



## **18. Representation Noising: A Defence Mechanism Against Harmful Finetuning**

cs.CL

Published in NeurIPs 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2405.14577v4) [paper-pdf](http://arxiv.org/pdf/2405.14577v4)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, David Atanasov, Robie Gonzales, Subhabrata Majumdar, Carsten Maple, Hassan Sajjad, Frank Rudzicz

**Abstract**: Releasing open-source large language models (LLMs) presents a dual-use risk since bad actors can easily fine-tune these models for harmful purposes. Even without the open release of weights, weight stealing and fine-tuning APIs make closed models vulnerable to harmful fine-tuning attacks (HFAs). While safety measures like preventing jailbreaks and improving safety guardrails are important, such measures can easily be reversed through fine-tuning. In this work, we propose Representation Noising (RepNoise), a defence mechanism that operates even when attackers have access to the weights. RepNoise works by removing information about harmful representations such that it is difficult to recover them during fine-tuning. Importantly, our defence is also able to generalize across different subsets of harm that have not been seen during the defence process as long as they are drawn from the same distribution of the attack set. Our method does not degrade the general capability of LLMs and retains the ability to train the model on harmless tasks. We provide empirical evidence that the efficacy of our defence lies in its ``depth'': the degree to which information about harmful representations is removed across all layers of the LLM. We also find areas where RepNoise still remains ineffective and highlight how those limitations can inform future research.



## **19. CARES: A Comprehensive Benchmark of Trustworthiness in Medical Vision Language Models**

cs.LG

NeurIPS 2024 Datasets and Benchmarks Track

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2406.06007v2) [paper-pdf](http://arxiv.org/pdf/2406.06007v2)

**Authors**: Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou, Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou, Kangyu Zhu, Wenhao Zheng, Zhaoyang Wang, Xiao Wang, Xuchao Zhang, Chetan Bansal, Marc Niethammer, Junzhou Huang, Hongtu Zhu, Yun Li, Jimeng Sun, Zongyuan Ge, Gang Li, James Zou, Huaxiu Yao

**Abstract**: Artificial intelligence has significantly impacted medical applications, particularly with the advent of Medical Large Vision Language Models (Med-LVLMs), sparking optimism for the future of automated and personalized healthcare. However, the trustworthiness of Med-LVLMs remains unverified, posing significant risks for future model deployment. In this paper, we introduce CARES and aim to comprehensively evaluate the Trustworthiness of Med-LVLMs across the medical domain. We assess the trustworthiness of Med-LVLMs across five dimensions, including trustfulness, fairness, safety, privacy, and robustness. CARES comprises about 41K question-answer pairs in both closed and open-ended formats, covering 16 medical image modalities and 27 anatomical regions. Our analysis reveals that the models consistently exhibit concerns regarding trustworthiness, often displaying factual inaccuracies and failing to maintain fairness across different demographic groups. Furthermore, they are vulnerable to attacks and demonstrate a lack of privacy awareness. We publicly release our benchmark and code in https://cares-ai.github.io/.



## **20. ProTransformer: Robustify Transformers via Plug-and-Play Paradigm**

cs.LG

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.23182v1) [paper-pdf](http://arxiv.org/pdf/2410.23182v1)

**Authors**: Zhichao Hou, Weizhi Gao, Yuchen Shen, Feiyi Wang, Xiaorui Liu

**Abstract**: Transformer-based architectures have dominated various areas of machine learning in recent years. In this paper, we introduce a novel robust attention mechanism designed to enhance the resilience of transformer-based architectures. Crucially, this technique can be integrated into existing transformers as a plug-and-play layer, improving their robustness without the need for additional training or fine-tuning. Through comprehensive experiments and ablation studies, we demonstrate that our ProTransformer significantly enhances the robustness of transformer models across a variety of prediction tasks, attack mechanisms, backbone architectures, and data domains. Notably, without further fine-tuning, the ProTransformer consistently improves the performance of vanilla transformers by 19.5%, 28.3%, 16.1%, and 11.4% for BERT, ALBERT, DistilBERT, and RoBERTa, respectively, under the classical TextFooler attack. Furthermore, ProTransformer shows promising resilience in large language models (LLMs) against prompting-based attacks, improving the performance of T5 and LLaMA by 24.8% and 17.8%, respectively, and enhancing Vicuna by an average of 10.4% against the Jailbreaking attack. Beyond the language domain, ProTransformer also demonstrates outstanding robustness in both vision and graph domains.



## **21. Effective and Efficient Adversarial Detection for Vision-Language Models via A Single Vector**

cs.CV

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22888v1) [paper-pdf](http://arxiv.org/pdf/2410.22888v1)

**Authors**: Youcheng Huang, Fengbin Zhu, Jingkun Tang, Pan Zhou, Wenqiang Lei, Jiancheng Lv, Tat-Seng Chua

**Abstract**: Visual Language Models (VLMs) are vulnerable to adversarial attacks, especially those from adversarial images, which is however under-explored in literature. To facilitate research on this critical safety problem, we first construct a new laRge-scale Adervsarial images dataset with Diverse hArmful Responses (RADAR), given that existing datasets are either small-scale or only contain limited types of harmful responses. With the new RADAR dataset, we further develop a novel and effective iN-time Embedding-based AdveRSarial Image DEtection (NEARSIDE) method, which exploits a single vector that distilled from the hidden states of VLMs, which we call the attacking direction, to achieve the detection of adversarial images against benign ones in the input. Extensive experiments with two victim VLMs, LLaVA and MiniGPT-4, well demonstrate the effectiveness, efficiency, and cross-model transferrability of our proposed method. Our code is available at https://github.com/mob-scu/RADAR-NEARSIDE



## **22. Stealth edits to large language models**

cs.AI

28 pages, 14 figures. Open source implementation:  https://github.com/qinghua-zhou/stealth-edits

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2406.12670v2) [paper-pdf](http://arxiv.org/pdf/2406.12670v2)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Wei Wang, Desmond J. Higham, Alexander N. Gorban, Alexander Bastounis, Ivan Y. Tyukin

**Abstract**: We reveal the theoretical foundations of techniques for editing large language models, and present new methods which can do so without requiring retraining. Our theoretical insights show that a single metric (a measure of the intrinsic dimension of the model's features) can be used to assess a model's editability and reveals its previously unrecognised susceptibility to malicious stealth attacks. This metric is fundamental to predicting the success of a variety of editing approaches, and reveals new bridges between disparate families of editing methods. We collectively refer to these as stealth editing methods, because they directly update a model's weights to specify its response to specific known hallucinating prompts without affecting other model behaviour. By carefully applying our theoretical insights, we are able to introduce a new jet-pack network block which is optimised for highly selective model editing, uses only standard network operations, and can be inserted into existing networks. We also reveal the vulnerability of language models to stealth attacks: a small change to a model's weights which fixes its response to a single attacker-chosen prompt. Stealth attacks are computationally simple, do not require access to or knowledge of the model's training data, and therefore represent a potent yet previously unrecognised threat to redistributed foundation models. Extensive experimental results illustrate and support our methods and their theoretical underpinnings. Demos and source code are available at https://github.com/qinghua-zhou/stealth-edits.



## **23. HijackRAG: Hijacking Attacks against Retrieval-Augmented Large Language Models**

cs.CR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22832v1) [paper-pdf](http://arxiv.org/pdf/2410.22832v1)

**Authors**: Yucheng Zhang, Qinfeng Li, Tianyu Du, Xuhong Zhang, Xinkui Zhao, Zhengwen Feng, Jianwei Yin

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge, making them adaptable and cost-effective for various applications. However, the growing reliance on these systems also introduces potential security risks. In this work, we reveal a novel vulnerability, the retrieval prompt hijack attack (HijackRAG), which enables attackers to manipulate the retrieval mechanisms of RAG systems by injecting malicious texts into the knowledge database. When the RAG system encounters target questions, it generates the attacker's pre-determined answers instead of the correct ones, undermining the integrity and trustworthiness of the system. We formalize HijackRAG as an optimization problem and propose both black-box and white-box attack strategies tailored to different levels of the attacker's knowledge. Extensive experiments on multiple benchmark datasets show that HijackRAG consistently achieves high attack success rates, outperforming existing baseline attacks. Furthermore, we demonstrate that the attack is transferable across different retriever models, underscoring the widespread risk it poses to RAG systems. Lastly, our exploration of various defense mechanisms reveals that they are insufficient to counter HijackRAG, emphasizing the urgent need for more robust security measures to protect RAG systems in real-world deployments.



## **24. InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models**

cs.CL

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22770v1) [paper-pdf](http://arxiv.org/pdf/2410.22770v1)

**Authors**: Hao Li, Xiaogeng Liu, Chaowei Xiao

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), enabling goal hijacking and data leakage. Prompt guard models, though effective in defense, suffer from over-defense -- falsely flagging benign inputs as malicious due to trigger word bias. To address this issue, we introduce NotInject, an evaluation dataset that systematically measures over-defense across various prompt guard models. NotInject contains 339 benign samples enriched with trigger words common in prompt injection attacks, enabling fine-grained evaluation. Our results show that state-of-the-art models suffer from over-defense issues, with accuracy dropping close to random guessing levels (60%). To mitigate this, we propose InjecGuard, a novel prompt guard model that incorporates a new training strategy, Mitigating Over-defense for Free (MOF), which significantly reduces the bias on trigger words. InjecGuard demonstrates state-of-the-art performance on diverse benchmarks including NotInject, surpassing the existing best model by 30.8%, offering a robust and open-source solution for detecting prompt injection attacks. The code and datasets are released at https://github.com/SaFoLab-WISC/InjecGuard.



## **25. Uncovering Coordinated Cross-Platform Information Operations Threatening the Integrity of the 2024 U.S. Presidential Election Online Discussion**

cs.SI

First Monday 29(11), 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2409.15402v2) [paper-pdf](http://arxiv.org/pdf/2409.15402v2)

**Authors**: Marco Minici, Luca Luceri, Federico Cinus, Emilio Ferrara

**Abstract**: Information Operations (IOs) pose a significant threat to the integrity of democratic processes, with the potential to influence election-related online discourse. In anticipation of the 2024 U.S. presidential election, we present a study aimed at uncovering the digital traces of coordinated IOs on $\mathbb{X}$ (formerly Twitter). Using our machine learning framework for detecting online coordination, we analyze a dataset comprising election-related conversations on $\mathbb{X}$ from May 2024. This reveals a network of coordinated inauthentic actors, displaying notable similarities in their link-sharing behaviors. Our analysis shows concerted efforts by these accounts to disseminate misleading, redundant, and biased information across the Web through a coordinated cross-platform information operation: The links shared by this network frequently direct users to other social media platforms or suspicious websites featuring low-quality political content and, in turn, promoting the same $\mathbb{X}$ and YouTube accounts. Members of this network also shared deceptive images generated by AI, accompanied by language attacking political figures and symbolic imagery intended to convey power and dominance. While $\mathbb{X}$ has suspended a subset of these accounts, more than 75% of the coordinated network remains active. Our findings underscore the critical role of developing computational models to scale up the detection of threats on large social media platforms, and emphasize the broader implications of these techniques to detect IOs across the wider Web.



## **26. SVIP: Towards Verifiable Inference of Open-source Large Language Models**

cs.LG

20 pages

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22307v1) [paper-pdf](http://arxiv.org/pdf/2410.22307v1)

**Authors**: Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, Huan Zhang

**Abstract**: Open-source Large Language Models (LLMs) have recently demonstrated remarkable capabilities in natural language understanding and generation, leading to widespread adoption across various domains. However, their increasing model sizes render local deployment impractical for individual users, pushing many to rely on computing service providers for inference through a blackbox API. This reliance introduces a new risk: a computing provider may stealthily substitute the requested LLM with a smaller, less capable model without consent from users, thereby delivering inferior outputs while benefiting from cost savings. In this paper, we formalize the problem of verifiable inference for LLMs. Existing verifiable computing solutions based on cryptographic or game-theoretic techniques are either computationally uneconomical or rest on strong assumptions. We introduce SVIP, a secret-based verifiable LLM inference protocol that leverages intermediate outputs from LLM as unique model identifiers. By training a proxy task on these outputs and requiring the computing provider to return both the generated text and the processed intermediate outputs, users can reliably verify whether the computing provider is acting honestly. In addition, the integration of a secret mechanism further enhances the security of our protocol. We thoroughly analyze our protocol under multiple strong and adaptive adversarial scenarios. Our extensive experiments demonstrate that SVIP is accurate, generalizable, computationally efficient, and resistant to various attacks. Notably, SVIP achieves false negative rates below 5% and false positive rates below 3%, while requiring less than 0.01 seconds per query for verification.



## **27. Embedding-based classifiers can detect prompt injection attacks**

cs.CR

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22284v1) [paper-pdf](http://arxiv.org/pdf/2410.22284v1)

**Authors**: Md. Ahsan Ayub, Subhabrata Majumdar

**Abstract**: Large Language Models (LLMs) are seeing significant adoption in every type of organization due to their exceptional generative capabilities. However, LLMs are found to be vulnerable to various adversarial attacks, particularly prompt injection attacks, which trick them into producing harmful or inappropriate content. Adversaries execute such attacks by crafting malicious prompts to deceive the LLMs. In this paper, we propose a novel approach based on embedding-based Machine Learning (ML) classifiers to protect LLM-based applications against this severe threat. We leverage three commonly used embedding models to generate embeddings of malicious and benign prompts and utilize ML classifiers to predict whether an input prompt is malicious. Out of several traditional ML methods, we achieve the best performance with classifiers built using Random Forest and XGBoost. Our classifiers outperform state-of-the-art prompt injection classifiers available in open-source implementations, which use encoder-only neural networks.



## **28. AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts**

cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22143v1) [paper-pdf](http://arxiv.org/pdf/2410.22143v1)

**Authors**: Vishal Kumar, Zeyi Liao, Jaylen Jones, Huan Sun

**Abstract**: Although large language models (LLMs) are typically aligned, they remain vulnerable to jailbreaking through either carefully crafted prompts in natural language or, interestingly, gibberish adversarial suffixes. However, gibberish tokens have received relatively less attention despite their success in attacking aligned LLMs. Recent work, AmpleGCG~\citep{liao2024amplegcg}, demonstrates that a generative model can quickly produce numerous customizable gibberish adversarial suffixes for any harmful query, exposing a range of alignment gaps in out-of-distribution (OOD) language spaces. To bring more attention to this area, we introduce AmpleGCG-Plus, an enhanced version that achieves better performance in fewer attempts. Through a series of exploratory experiments, we identify several training strategies to improve the learning of gibberish suffixes. Our results, verified under a strict evaluation setting, show that it outperforms AmpleGCG on both open-weight and closed-source models, achieving increases in attack success rate (ASR) of up to 17\% in the white-box setting against Llama-2-7B-chat, and more than tripling ASR in the black-box setting against GPT-4. Notably, AmpleGCG-Plus jailbreaks the newer GPT-4o series of models at similar rates to GPT-4, and, uncovers vulnerabilities against the recently proposed circuit breakers defense. We publicly release AmpleGCG-Plus along with our collected training datasets.



## **29. Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents**

cs.CR

Accepted at NeurIPS 2024, camera ready version. Code and data are  available at https://github.com/lancopku/agent-backdoor-attacks

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2402.11208v2) [paper-pdf](http://arxiv.org/pdf/2402.11208v2)

**Authors**: Wenkai Yang, Xiaohan Bi, Yankai Lin, Sishuo Chen, Jie Zhou, Xu Sun

**Abstract**: Driven by the rapid development of Large Language Models (LLMs), LLM-based agents have been developed to handle various real-world applications, including finance, healthcare, and shopping, etc. It is crucial to ensure the reliability and security of LLM-based agents during applications. However, the safety issues of LLM-based agents are currently under-explored. In this work, we take the first step to investigate one of the typical safety threats, backdoor attack, to LLM-based agents. We first formulate a general framework of agent backdoor attacks, then we present a thorough analysis of different forms of agent backdoor attacks. Specifically, compared with traditional backdoor attacks on LLMs that are only able to manipulate the user inputs and model outputs, agent backdoor attacks exhibit more diverse and covert forms: (1) From the perspective of the final attacking outcomes, the agent backdoor attacker can not only choose to manipulate the final output distribution, but also introduce the malicious behavior in an intermediate reasoning step only, while keeping the final output correct. (2) Furthermore, the former category can be divided into two subcategories based on trigger locations, in which the backdoor trigger can either be hidden in the user query or appear in an intermediate observation returned by the external environment. We implement the above variations of agent backdoor attacks on two typical agent tasks including web shopping and tool utilization. Extensive experiments show that LLM-based agents suffer severely from backdoor attacks and such backdoor vulnerability cannot be easily mitigated by current textual backdoor defense algorithms. This indicates an urgent need for further research on the development of targeted defenses against backdoor attacks on LLM-based agents. Warning: This paper may contain biased content.



## **30. Waterfall: Framework for Robust and Scalable Text Watermarking and Provenance for LLMs**

cs.CR

Accepted to EMNLP 2024 Main Conference

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2407.04411v2) [paper-pdf](http://arxiv.org/pdf/2407.04411v2)

**Authors**: Gregory Kang Ruey Lau, Xinyuan Niu, Hieu Dao, Jiangwei Chen, Chuan-Sheng Foo, Bryan Kian Hsiang Low

**Abstract**: Protecting intellectual property (IP) of text such as articles and code is increasingly important, especially as sophisticated attacks become possible, such as paraphrasing by large language models (LLMs) or even unauthorized training of LLMs on copyrighted text to infringe such IP. However, existing text watermarking methods are not robust enough against such attacks nor scalable to millions of users for practical implementation. In this paper, we propose Waterfall, the first training-free framework for robust and scalable text watermarking applicable across multiple text types (e.g., articles, code) and languages supportable by LLMs, for general text and LLM data provenance. Waterfall comprises several key innovations, such as being the first to use LLM as paraphrasers for watermarking along with a novel combination of techniques that are surprisingly effective in achieving robust verifiability and scalability. We empirically demonstrate that Waterfall achieves significantly better scalability, robust verifiability, and computational efficiency compared to SOTA article-text watermarking methods, and showed how it could be directly applied to the watermarking of code. We also demonstrated that Waterfall can be used for LLM data provenance, where the watermarks of LLM training data can be detected in LLM output, allowing for detection of unauthorized use of data for LLM training and potentially enabling model-centric watermarking of open-sourced LLMs which has been a limitation of existing LLM watermarking works. Our code is available at https://github.com/aoi3142/Waterfall.



## **31. Enhancing Adversarial Attacks through Chain of Thought**

cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21791v1) [paper-pdf](http://arxiv.org/pdf/2410.21791v1)

**Authors**: Jingbo Su

**Abstract**: Large language models (LLMs) have demonstrated impressive performance across various domains but remain susceptible to safety concerns. Prior research indicates that gradient-based adversarial attacks are particularly effective against aligned LLMs and the chain of thought (CoT) prompting can elicit desired answers through step-by-step reasoning. This paper proposes enhancing the robustness of adversarial attacks on aligned LLMs by integrating CoT prompts with the greedy coordinate gradient (GCG) technique. Using CoT triggers instead of affirmative targets stimulates the reasoning abilities of backend LLMs, thereby improving the transferability and universality of adversarial attacks. We conducted an ablation study comparing our CoT-GCG approach with Amazon Web Services auto-cot. Results revealed our approach outperformed both the baseline GCG attack and CoT prompting. Additionally, we used Llama Guard to evaluate potentially harmful interactions, providing a more objective risk assessment of entire conversations compared to matching outputs to rejection phrases. The code of this paper is available at https://github.com/sujingbo0217/CS222W24-LLM-Attack.



## **32. Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning Attack**

cs.LG

Rejected by ICML2024. Accepted by NeurIPS2024

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2402.01109v5) [paper-pdf](http://arxiv.org/pdf/2402.01109v5)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.



## **33. Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2409.18169v4) [paper-pdf](http://arxiv.org/pdf/2409.18169v4)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent research demonstrates that the nascent fine-tuning-as-a-service business model exposes serious safety concerns -- fine-tuning over a few harmful data uploaded by the users can compromise the safety alignment of the model. The attack, known as harmful fine-tuning, has raised a broad research interest among the community. However, as the attack is still new, \textbf{we observe from our miserable submission experience that there are general misunderstandings within the research community.} We in this paper aim to clear some common concerns for the attack setting, and formally establish the research problem. Specifically, we first present the threat model of the problem, and introduce the harmful fine-tuning attack and its variants. Then we systematically survey the existing literature on attacks/defenses/mechanical analysis of the problem. Finally, we outline future research directions that might contribute to the development of the field. Additionally, we present a list of questions of interest, which might be useful to refer to when reviewers in the peer review process question the realism of the experiment/attack/defense setting. A curated list of relevant papers is maintained and made accessible at: \url{https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers}.



## **34. Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack**

cs.LG

Accepted by NeurIPS2024. arXiv admin note: substantial text overlap  with arXiv:2402.01109

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2405.18641v5) [paper-pdf](http://arxiv.org/pdf/2405.18641v5)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent studies show that Large Language Models (LLMs) with safety alignment can be jail-broken by fine-tuning on a dataset mixed with harmful data. First time in the literature, we show that the jail-broken effect can be mitigated by separating states in the finetuning stage to optimize the alignment and user datasets. Unfortunately, our subsequent study shows that this simple Bi-State Optimization (BSO) solution experiences convergence instability when steps invested in its alignment state is too small, leading to downgraded alignment performance. By statistical analysis, we show that the \textit{excess drift} towards consensus could be a probable reason for the instability. To remedy this issue, we propose \textbf{L}azy(\textbf{i}) \textbf{s}afety \textbf{a}lignment (\textbf{Lisa}), which introduces a proximal term to constraint the drift of each state. Theoretically, the benefit of the proximal term is supported by the convergence analysis, wherein we show that a sufficient large proximal factor is necessary to guarantee Lisa's convergence. Empirically, our results on four downstream finetuning tasks show that Lisa with a proximal term can significantly increase alignment performance while maintaining the LLM's accuracy on the user tasks. Code is available at \url{https://github.com/git-disl/Lisa}.



## **35. Fine-tuning Large Language Models for DGA and DNS Exfiltration Detection**

cs.CR

Accepted in Proceedings of the Workshop at AI for Cyber Threat  Intelligence (WAITI), 2024

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21723v1) [paper-pdf](http://arxiv.org/pdf/2410.21723v1)

**Authors**: Md Abu Sayed, Asif Rahman, Christopher Kiekintveld, Sebastian Garcia

**Abstract**: Domain Generation Algorithms (DGAs) are malicious techniques used by malware to dynamically generate seemingly random domain names for communication with Command & Control (C&C) servers. Due to the fast and simple generation of DGA domains, detection methods must be highly efficient and precise to be effective. Large Language Models (LLMs) have demonstrated their proficiency in real-time detection tasks, making them ideal candidates for detecting DGAs. Our work validates the effectiveness of fine-tuned LLMs for detecting DGAs and DNS exfiltration attacks. We developed LLM models and conducted comprehensive evaluation using a diverse dataset comprising 59 distinct real-world DGA malware families and normal domain data. Our LLM model significantly outperformed traditional natural language processing techniques, especially in detecting unknown DGAs. We also evaluated its performance on DNS exfiltration datasets, demonstrating its effectiveness in enhancing cybersecurity measures. To the best of our knowledge, this is the first work that empirically applies LLMs for DGA and DNS exfiltration detection.



## **36. Bileve: Securing Text Provenance in Large Language Models Against Spoofing with Bi-level Signature**

cs.CR

NeurIPS 2024 camera-ready

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2406.01946v3) [paper-pdf](http://arxiv.org/pdf/2406.01946v3)

**Authors**: Tong Zhou, Xuandong Zhao, Xiaolin Xu, Shaolei Ren

**Abstract**: Text watermarks for large language models (LLMs) have been commonly used to identify the origins of machine-generated content, which is promising for assessing liability when combating deepfake or harmful content. While existing watermarking techniques typically prioritize robustness against removal attacks, unfortunately, they are vulnerable to spoofing attacks: malicious actors can subtly alter the meanings of LLM-generated responses or even forge harmful content, potentially misattributing blame to the LLM developer. To overcome this, we introduce a bi-level signature scheme, Bileve, which embeds fine-grained signature bits for integrity checks (mitigating spoofing attacks) as well as a coarse-grained signal to trace text sources when the signature is invalid (enhancing detectability) via a novel rank-based sampling strategy. Compared to conventional watermark detectors that only output binary results, Bileve can differentiate 5 scenarios during detection, reliably tracing text provenance and regulating LLMs. The experiments conducted on OPT-1.3B and LLaMA-7B demonstrate the effectiveness of Bileve in defeating spoofing attacks with enhanced detectability. Code is available at https://github.com/Tongzhou0101/Bileve-official.



## **37. CFSafety: Comprehensive Fine-grained Safety Assessment for LLMs**

cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21695v1) [paper-pdf](http://arxiv.org/pdf/2410.21695v1)

**Authors**: Zhihao Liu, Chenhui Hu

**Abstract**: As large language models (LLMs) rapidly evolve, they bring significant conveniences to our work and daily lives, but also introduce considerable safety risks. These models can generate texts with social biases or unethical content, and under specific adversarial instructions, may even incite illegal activities. Therefore, rigorous safety assessments of LLMs are crucial. In this work, we introduce a safety assessment benchmark, CFSafety, which integrates 5 classic safety scenarios and 5 types of instruction attacks, totaling 10 categories of safety questions, to form a test set with 25k prompts. This test set was used to evaluate the natural language generation (NLG) capabilities of LLMs, employing a combination of simple moral judgment and a 1-5 safety rating scale for scoring. Using this benchmark, we tested eight popular LLMs, including the GPT series. The results indicate that while GPT-4 demonstrated superior safety performance, the safety effectiveness of LLMs, including this model, still requires improvement. The data and code associated with this study are available on GitHub.



## **38. FATH: Authentication-based Test-time Defense against Indirect Prompt Injection Attacks**

cs.CR

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21492v1) [paper-pdf](http://arxiv.org/pdf/2410.21492v1)

**Authors**: Jiongxiao Wang, Fangzhou Wu, Wendi Li, Jinsheng Pan, Edward Suh, Z. Morley Mao, Muhao Chen, Chaowei Xiao

**Abstract**: Large language models (LLMs) have been widely deployed as the backbone with additional tools and text information for real-world applications. However, integrating external information into LLM-integrated applications raises significant security concerns. Among these, prompt injection attacks are particularly threatening, where malicious instructions injected in the external text information can exploit LLMs to generate answers as the attackers desire. While both training-time and test-time defense methods have been developed to mitigate such attacks, the unaffordable training costs associated with training-time methods and the limited effectiveness of existing test-time methods make them impractical. This paper introduces a novel test-time defense strategy, named Formatting AuThentication with Hash-based tags (FATH). Unlike existing approaches that prevent LLMs from answering additional instructions in external text, our method implements an authentication system, requiring LLMs to answer all received instructions with a security policy and selectively filter out responses to user instructions as the final output. To achieve this, we utilize hash-based authentication tags to label each response, facilitating accurate identification of responses according to the user's instructions and improving the robustness against adaptive attacks. Comprehensive experiments demonstrate that our defense method can effectively defend against indirect prompt injection attacks, achieving state-of-the-art performance under Llama3 and GPT3.5 models across various attack methods. Our code is released at: https://github.com/Jayfeather1024/FATH



## **39. Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning**

cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.07163v2) [paper-pdf](http://arxiv.org/pdf/2410.07163v2)

**Authors**: Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, Sijia Liu

**Abstract**: In this work, we address the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences and associated model capabilities (e.g., copyrighted data or harmful content generation) while preserving essential model utilities, without the need for retraining from scratch. Despite the growing need for LLM unlearning, a principled optimization framework remains lacking. To this end, we revisit the state-of-the-art approach, negative preference optimization (NPO), and identify the issue of reference model bias, which could undermine NPO's effectiveness, particularly when unlearning forget data of varying difficulty. Given that, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that 'simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We also provide deeper insights into SimNPO's advantages, supported by analysis using mixtures of Markov chains. Furthermore, we present extensive experiments validating SimNPO's superiority over existing unlearning baselines in benchmarks like TOFU and MUSE, and robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.



## **40. Systematically Analyzing Prompt Injection Vulnerabilities in Diverse LLM Architectures**

cs.CR

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.23308v1) [paper-pdf](http://arxiv.org/pdf/2410.23308v1)

**Authors**: Victoria Benjamin, Emily Braca, Israel Carter, Hafsa Kanchwala, Nava Khojasteh, Charly Landow, Yi Luo, Caroline Ma, Anna Magarelli, Rachel Mirin, Avery Moyer, Kayla Simpson, Amelia Skawinski, Thomas Heverin

**Abstract**: This study systematically analyzes the vulnerability of 36 large language models (LLMs) to various prompt injection attacks, a technique that leverages carefully crafted prompts to elicit malicious LLM behavior. Across 144 prompt injection tests, we observed a strong correlation between model parameters and vulnerability, with statistical analyses, such as logistic regression and random forest feature analysis, indicating that parameter size and architecture significantly influence susceptibility. Results revealed that 56 percent of tests led to successful prompt injections, emphasizing widespread vulnerability across various parameter sizes, with clustering analysis identifying distinct vulnerability profiles associated with specific model configurations. Additionally, our analysis uncovered correlations between certain prompt injection techniques, suggesting potential overlaps in vulnerabilities. These findings underscore the urgent need for robust, multi-layered defenses in LLMs deployed across critical infrastructure and sensitive industries. Successful prompt injection attacks could result in severe consequences, including data breaches, unauthorized access, or misinformation. Future research should explore multilingual and multi-step defenses alongside adaptive mitigation strategies to strengthen LLM security in diverse, real-world environments.



## **41. Securing Multi-turn Conversational Language Models From Distributed Backdoor Triggers**

cs.CL

Findings of EMNLP 2024

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2407.04151v2) [paper-pdf](http://arxiv.org/pdf/2407.04151v2)

**Authors**: Terry Tong, Jiashu Xu, Qin Liu, Muhao Chen

**Abstract**: Large language models (LLMs) have acquired the ability to handle longer context lengths and understand nuances in text, expanding their dialogue capabilities beyond a single utterance. A popular user-facing application of LLMs is the multi-turn chat setting. Though longer chat memory and better understanding may seemingly benefit users, our paper exposes a vulnerability that leverages the multi-turn feature and strong learning ability of LLMs to harm the end-user: the backdoor. We demonstrate that LLMs can capture the combinational backdoor representation. Only upon presentation of triggers together does the backdoor activate. We also verify empirically that this representation is invariant to the position of the trigger utterance. Subsequently, inserting a single extra token into two utterances of 5%of the data can cause over 99% Attack Success Rate (ASR). Our results with 3 triggers demonstrate that this framework is generalizable, compatible with any trigger in an adversary's toolbox in a plug-and-play manner. Defending the backdoor can be challenging in the chat setting because of the large input and output space. Our analysis indicates that the distributed backdoor exacerbates the current challenges by polynomially increasing the dimension of the attacked input space. Canonical textual defenses like ONION and BKI leverage auxiliary model forward passes over individual tokens, scaling exponentially with the input sequence length and struggling to maintain computational feasibility. To this end, we propose a decoding time defense - decayed contrastive decoding - that scales linearly with assistant response sequence length and reduces the backdoor to as low as 0.35%.



## **42. AutoPenBench: Benchmarking Generative Agents for Penetration Testing**

cs.CR

Codes for the benchmark:  https://github.com/lucagioacchini/auto-pen-bench Codes for the paper  experiments: https://github.com/lucagioacchini/genai-pentest-paper

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.03225v2) [paper-pdf](http://arxiv.org/pdf/2410.03225v2)

**Authors**: Luca Gioacchini, Marco Mellia, Idilio Drago, Alexander Delsanto, Giuseppe Siracusano, Roberto Bifulco

**Abstract**: Generative AI agents, software systems powered by Large Language Models (LLMs), are emerging as a promising approach to automate cybersecurity tasks. Among the others, penetration testing is a challenging field due to the task complexity and the diverse strategies to simulate cyber-attacks. Despite growing interest and initial studies in automating penetration testing with generative agents, there remains a significant gap in the form of a comprehensive and standard framework for their evaluation and development. This paper introduces AutoPenBench, an open benchmark for evaluating generative agents in automated penetration testing. We present a comprehensive framework that includes 33 tasks, each representing a vulnerable system that the agent has to attack. Tasks are of increasing difficulty levels, including in-vitro and real-world scenarios. We assess the agent performance with generic and specific milestones that allow us to compare results in a standardised manner and understand the limits of the agent under test. We show the benefits of AutoPenBench by testing two agent architectures: a fully autonomous and a semi-autonomous supporting human interaction. We compare their performance and limitations. For example, the fully autonomous agent performs unsatisfactorily achieving a 21% Success Rate (SR) across the benchmark, solving 27% of the simple tasks and only one real-world task. In contrast, the assisted agent demonstrates substantial improvements, with 64% of SR. AutoPenBench allows us also to observe how different LLMs like GPT-4o or OpenAI o1 impact the ability of the agents to complete the tasks. We believe that our benchmark fills the gap with a standard and flexible framework to compare penetration testing agents on a common ground. We hope to extend AutoPenBench along with the research community by making it available under https://github.com/lucagioacchini/auto-pen-bench.



## **43. Palisade -- Prompt Injection Detection Framework**

cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21146v1) [paper-pdf](http://arxiv.org/pdf/2410.21146v1)

**Authors**: Sahasra Kokkula, Somanathan R, Nandavardhan R, Aashishkumar, G Divya

**Abstract**: The advent of Large Language Models LLMs marks a milestone in Artificial Intelligence, altering how machines comprehend and generate human language. However, LLMs are vulnerable to malicious prompt injection attacks, where crafted inputs manipulate the models behavior in unintended ways, compromising system integrity and causing incorrect outcomes. Conventional detection methods rely on static, rule-based approaches, which often fail against sophisticated threats like abnormal token sequences and alias substitutions, leading to limited adaptability and higher rates of false positives and false negatives.This paper proposes a novel NLP based approach for prompt injection detection, emphasizing accuracy and optimization through a layered input screening process. In this framework, prompts are filtered through three distinct layers rule-based, ML classifier, and companion LLM before reaching the target model, thereby minimizing the risk of malicious interaction.Tests show the ML classifier achieves the highest accuracy among individual layers, yet the multi-layer framework enhances overall detection accuracy by reducing false negatives. Although this increases false positives, it minimizes the risk of overlooking genuine injected prompts, thus prioritizing security.This multi-layered detection approach highlights LLM vulnerabilities and provides a comprehensive framework for future research, promoting secure interactions between humans and AI systems.



## **44. Stealthy Jailbreak Attacks on Large Language Models via Benign Data Mirroring**

cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21083v1) [paper-pdf](http://arxiv.org/pdf/2410.21083v1)

**Authors**: Honglin Mu, Han He, Yuxin Zhou, Yunlong Feng, Yang Xu, Libo Qin, Xiaoming Shi, Zeming Liu, Xudong Han, Qi Shi, Qingfu Zhu, Wanxiang Che

**Abstract**: Large language model (LLM) safety is a critical issue, with numerous studies employing red team testing to enhance model security. Among these, jailbreak methods explore potential vulnerabilities by crafting malicious prompts that induce model outputs contrary to safety alignments. Existing black-box jailbreak methods often rely on model feedback, repeatedly submitting queries with detectable malicious instructions during the attack search process. Although these approaches are effective, the attacks may be intercepted by content moderators during the search process. We propose an improved transfer attack method that guides malicious prompt construction by locally training a mirror model of the target black-box model through benign data distillation. This method offers enhanced stealth, as it does not involve submitting identifiable malicious instructions to the target model during the search phase. Our approach achieved a maximum attack success rate of 92%, or a balanced value of 80% with an average of 1.5 detectable jailbreak queries per sample against GPT-3.5 Turbo on a subset of AdvBench. These results underscore the need for more robust defense mechanisms.



## **45. Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models**

cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.20940v1) [paper-pdf](http://arxiv.org/pdf/2410.20940v1)

**Authors**: Piotr Przybyła

**Abstract**: We investigate the challenge of generating adversarial examples to test the robustness of text classification algorithms detecting low-credibility content, including propaganda, false claims, rumours and hyperpartisan news. We focus on simulation of content moderation by setting realistic limits on the number of queries an attacker is allowed to attempt. Within our solution (TREPAT), initial rephrasings are generated by large language models with prompts inspired by meaning-preserving NLP tasks, e.g. text simplification and style transfer. Subsequently, these modifications are decomposed into small changes, applied through beam search procedure until the victim classifier changes its decision. The evaluation confirms the superiority of our approach in the constrained scenario, especially in case of long input text (news articles), where exhaustive search is not feasible.



## **46. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

cs.CR

v0.1

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.20911v1) [paper-pdf](http://arxiv.org/pdf/2410.20911v1)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis



## **47. Uncovering Safety Risks of Large Language Models through Concept Activation Vector**

cs.CL

10 pages, accepted as a poster at NeurIPS 2024

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2404.12038v4) [paper-pdf](http://arxiv.org/pdf/2404.12038v4)

**Authors**: Zhihao Xu, Ruixuan Huang, Changyu Chen, Xiting Wang

**Abstract**: Despite careful safety alignment, current large language models (LLMs) remain vulnerable to various attacks. To further unveil the safety risks of LLMs, we introduce a Safety Concept Activation Vector (SCAV) framework, which effectively guides the attacks by accurately interpreting LLMs' safety mechanisms. We then develop an SCAV-guided attack method that can generate both attack prompts and embedding-level attacks with automatically selected perturbation hyperparameters. Both automatic and human evaluations demonstrate that our attack method significantly improves the attack success rate and response quality while requiring less training data. Additionally, we find that our generated attack prompts may be transferable to GPT-4, and the embedding-level attacks may also be transferred to other white-box LLMs whose parameters are known. Our experiments further uncover the safety risks present in current LLMs. For example, in our evaluation of seven open-source LLMs, we observe an average attack success rate of 99.14%, based on the classic keyword-matching criterion. Finally, we provide insights into the safety mechanism of LLMs. The code is available at https://github.com/SproutNan/AI-Safety_SCAV.



## **48. Fine-tuned Large Language Models (LLMs): Improved Prompt Injection Attacks Detection**

cs.CL

**SubmitDate**: 2024-10-28    [abs](http://arxiv.org/abs/2410.21337v1) [paper-pdf](http://arxiv.org/pdf/2410.21337v1)

**Authors**: Md Abdur Rahman, Fan Wu, Alfredo Cuzzocrea, Sheikh Iqbal Ahamed

**Abstract**: Large language models (LLMs) are becoming a popular tool as they have significantly advanced in their capability to tackle a wide range of language-based tasks. However, LLMs applications are highly vulnerable to prompt injection attacks, which poses a critical problem. These attacks target LLMs applications through using carefully designed input prompts to divert the model from adhering to original instruction, thereby it could execute unintended actions. These manipulations pose serious security threats which potentially results in data leaks, biased outputs, or harmful responses. This project explores the security vulnerabilities in relation to prompt injection attacks. To detect whether a prompt is vulnerable or not, we follows two approaches: 1) a pre-trained LLM, and 2) a fine-tuned LLM. Then, we conduct a thorough analysis and comparison of the classification performance. Firstly, we use pre-trained XLM-RoBERTa model to detect prompt injections using test dataset without any fine-tuning and evaluate it by zero-shot classification. Then, this proposed work will apply supervised fine-tuning to this pre-trained LLM using a task-specific labeled dataset from deepset in huggingface, and this fine-tuned model achieves impressive results with 99.13\% accuracy, 100\% precision, 98.33\% recall and 99.15\% F1-score thorough rigorous experimentation and evaluation. We observe that our approach is highly efficient in detecting prompt injection attacks.



## **49. LLM Robustness Against Misinformation in Biomedical Question Answering**

cs.CL

**SubmitDate**: 2024-10-27    [abs](http://arxiv.org/abs/2410.21330v1) [paper-pdf](http://arxiv.org/pdf/2410.21330v1)

**Authors**: Alexander Bondarenko, Adrian Viehweger

**Abstract**: The retrieval-augmented generation (RAG) approach is used to reduce the confabulation of large language models (LLMs) for question answering by retrieving and providing additional context coming from external knowledge sources (e.g., by adding the context to the prompt). However, injecting incorrect information can mislead the LLM to generate an incorrect answer.   In this paper, we evaluate the effectiveness and robustness of four LLMs against misinformation - Gemma 2, GPT-4o-mini, Llama~3.1, and Mixtral - in answering biomedical questions. We assess the answer accuracy on yes-no and free-form questions in three scenarios: vanilla LLM answers (no context is provided), "perfect" augmented generation (correct context is provided), and prompt-injection attacks (incorrect context is provided). Our results show that Llama 3.1 (70B parameters) achieves the highest accuracy in both vanilla (0.651) and "perfect" RAG (0.802) scenarios. However, the accuracy gap between the models almost disappears with "perfect" RAG, suggesting its potential to mitigate the LLM's size-related effectiveness differences.   We further evaluate the ability of the LLMs to generate malicious context on one hand and the LLM's robustness against prompt-injection attacks on the other hand, using metrics such as attack success rate (ASR), accuracy under attack, and accuracy drop. As adversaries, we use the same four LLMs (Gemma 2, GPT-4o-mini, Llama 3.1, and Mixtral) to generate incorrect context that is injected in the target model's prompt. Interestingly, Llama is shown to be the most effective adversary, causing accuracy drops of up to 0.48 for vanilla answers and 0.63 for "perfect" RAG across target models. Our analysis reveals that robustness rankings vary depending on the evaluation measure, highlighting the complexity of assessing LLM resilience to adversarial attacks.



## **50. PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining**

cs.CR

36 pages

**SubmitDate**: 2024-10-26    [abs](http://arxiv.org/abs/2402.09477v2) [paper-pdf](http://arxiv.org/pdf/2402.09477v2)

**Authors**: Mishaal Kazmi, Hadrien Lautraite, Alireza Akbari, Qiaoyue Tang, Mauricio Soroco, Tao Wang, Sébastien Gambs, Mathias Lécuyer

**Abstract**: We present PANORAMIA, a privacy leakage measurement framework for machine learning models that relies on membership inference attacks using generated data as non-members. By relying on generated non-member data, PANORAMIA eliminates the common dependency of privacy measurement tools on in-distribution non-member data. As a result, PANORAMIA does not modify the model, training data, or training process, and only requires access to a subset of the training data. We evaluate PANORAMIA on ML models for image and tabular data classification, as well as on large-scale language models.



