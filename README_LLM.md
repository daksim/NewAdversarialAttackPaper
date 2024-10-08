# Latest Large Language Model Attack Papers
**update at 2024-10-08 09:45:24**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Obliviate: Neutralizing Task-agnostic Backdoors within the Parameter-efficient Fine-tuning Paradigm**

cs.CL

Under Review

**SubmitDate**: 2024-10-06    [abs](http://arxiv.org/abs/2409.14119v3) [paper-pdf](http://arxiv.org/pdf/2409.14119v3)

**Authors**: Jaehan Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin

**Abstract**: Parameter-efficient fine-tuning (PEFT) has become a key training strategy for large language models. However, its reliance on fewer trainable parameters poses security risks, such as task-agnostic backdoors. Despite their severe impact on a wide range of tasks, there is no practical defense solution available that effectively counters task-agnostic backdoors within the context of PEFT. In this study, we introduce Obliviate, a PEFT-integrable backdoor defense. We develop two techniques aimed at amplifying benign neurons within PEFT layers and penalizing the influence of trigger tokens. Our evaluations across three major PEFT architectures show that our method can significantly reduce the attack success rate of the state-of-the-art task-agnostic backdoors (83.6%$\downarrow$). Furthermore, our method exhibits robust defense capabilities against both task-specific backdoors and adaptive attacks. Source code will be obtained at https://github.com/obliviateARR/Obliviate.



## **2. AppPoet: Large Language Model based Android malware detection via multi-view prompt engineering**

cs.CR

**SubmitDate**: 2024-10-06    [abs](http://arxiv.org/abs/2404.18816v2) [paper-pdf](http://arxiv.org/pdf/2404.18816v2)

**Authors**: Wenxiang Zhao, Juntao Wu, Zhaoyi Meng

**Abstract**: Due to the vast array of Android applications, their multifarious functions and intricate behavioral semantics, attackers can adopt various tactics to conceal their genuine attack intentions within legitimate functions. However, numerous learning-based methods suffer from a limitation in mining behavioral semantic information, thus impeding the accuracy and efficiency of Android malware detection. Besides, the majority of existing learning-based methods are weakly interpretive and fail to furnish researchers with effective and readable detection reports. Inspired by the success of the Large Language Models (LLMs) in natural language understanding, we propose AppPoet, a LLM-assisted multi-view system for Android malware detection. Firstly, AppPoet employs a static method to comprehensively collect application features and formulate various observation views. Then, using our carefully crafted multi-view prompt templates, it guides the LLM to generate function descriptions and behavioral summaries for each view, enabling deep semantic analysis of the views. Finally, we collaboratively fuse the multi-view information to efficiently and accurately detect malware through a deep neural network (DNN) classifier and then generate the human-readable diagnostic reports. Experimental results demonstrate that our method achieves a detection accuracy of 97.15% and an F1 score of 97.21%, which is superior to the baseline methods. Furthermore, the case study evaluates the effectiveness of our generated diagnostic reports.



## **3. Functional Homotopy: Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Attacks**

cs.LG

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04234v1) [paper-pdf](http://arxiv.org/pdf/2410.04234v1)

**Authors**: Zi Wang, Divyam Anshumaan, Ashish Hooda, Yudong Chen, Somesh Jha

**Abstract**: Optimization methods are widely employed in deep learning to identify and mitigate undesired model responses. While gradient-based techniques have proven effective for image models, their application to language models is hindered by the discrete nature of the input space. This study introduces a novel optimization approach, termed the \emph{functional homotopy} method, which leverages the functional duality between model training and input generation. By constructing a series of easy-to-hard optimization problems, we iteratively solve these problems using principles derived from established homotopy methods. We apply this approach to jailbreak attack synthesis for large language models (LLMs), achieving a $20\%-30\%$ improvement in success rate over existing methods in circumventing established safe open-source models such as Llama-2 and Llama-3.



## **4. Adversarial Suffixes May Be Features Too!**

cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.00451v2) [paper-pdf](http://arxiv.org/pdf/2410.00451v2)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Despite significant ongoing efforts in safety alignment, large language models (LLMs) such as GPT-4 and LLaMA 3 remain vulnerable to jailbreak attacks that can induce harmful behaviors, including those triggered by adversarial suffixes. Building on prior research, we hypothesize that these adversarial suffixes are not mere bugs but may represent features that can dominate the LLM's behavior. To evaluate this hypothesis, we conduct several experiments. First, we demonstrate that benign features can be effectively made to function as adversarial suffixes, i.e., we develop a feature extraction method to extract sample-agnostic features from benign dataset in the form of suffixes and show that these suffixes may effectively compromise safety alignment. Second, we show that adversarial suffixes generated from jailbreak attacks may contain meaningful features, i.e., appending the same suffix to different prompts results in responses exhibiting specific characteristics. Third, we show that such benign-yet-safety-compromising features can be easily introduced through fine-tuning using only benign datasets, i.e., even in the absence of harmful content. This highlights the critical risk posed by dominating benign features in the training data and calls for further research to reinforce LLM safety alignment. Our code and data is available at \url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.



## **5. Automated Progressive Red Teaming**

cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2407.03876v2) [paper-pdf](http://arxiv.org/pdf/2407.03876v2)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong

**Abstract**: Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming (ART) offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current ART efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of ARPT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access), and 39% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).



## **6. Harnessing Task Overload for Scalable Jailbreak Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04190v1) [paper-pdf](http://arxiv.org/pdf/2410.04190v1)

**Authors**: Yiting Dong, Guobin Shen, Dongcheng Zhao, Xiang He, Yi Zeng

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreak attacks that bypass their safety mechanisms. Existing attack methods are fixed or specifically tailored for certain models and cannot flexibly adjust attack strength, which is critical for generalization when attacking models of various sizes. We introduce a novel scalable jailbreak attack that preempts the activation of an LLM's safety policies by occupying its computational resources. Our method involves engaging the LLM in a resource-intensive preliminary task - a Character Map lookup and decoding process - before presenting the target instruction. By saturating the model's processing capacity, we prevent the activation of safety protocols when processing the subsequent instruction. Extensive experiments on state-of-the-art LLMs demonstrate that our method achieves a high success rate in bypassing safety measures without requiring gradient access, manual prompt engineering. We verified our approach offers a scalable attack that quantifies attack strength and adapts to different model scales at the optimal strength. We shows safety policies of LLMs might be more susceptible to resource constraints. Our findings reveal a critical vulnerability in current LLM safety designs, highlighting the need for more robust defense strategies that account for resource-intense condition.



## **7. Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-based Decision-Making Systems**

cs.CR

31 pages, including main paper, references, and appendix

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2405.20774v2) [paper-pdf](http://arxiv.org/pdf/2405.20774v2)

**Authors**: Ruochen Jiao, Shaoyuan Xie, Justin Yue, Takami Sato, Lixu Wang, Yixuan Wang, Qi Alfred Chen, Qi Zhu

**Abstract**: Large Language Models (LLMs) have shown significant promise in real-world decision-making tasks for embodied artificial intelligence, especially when fine-tuned to leverage their inherent common sense and reasoning abilities while being tailored to specific applications. However, this fine-tuning process introduces considerable safety and security vulnerabilities, especially in safety-critical cyber-physical systems. In this work, we propose the first comprehensive framework for Backdoor Attacks against LLM-based Decision-making systems (BALD) in embodied AI, systematically exploring the attack surfaces and trigger mechanisms. Specifically, we propose three distinct attack mechanisms: word injection, scenario manipulation, and knowledge injection, targeting various components in the LLM-based decision-making pipeline. We perform extensive experiments on representative LLMs (GPT-3.5, LLaMA2, PaLM2) in autonomous driving and home robot tasks, demonstrating the effectiveness and stealthiness of our backdoor triggers across various attack channels, with cases like vehicles accelerating toward obstacles and robots placing knives on beds. Our word and knowledge injection attacks achieve nearly 100% success rate across multiple models and datasets while requiring only limited access to the system. Our scenario manipulation attack yields success rates exceeding 65%, reaching up to 90%, and does not require any runtime system intrusion. We also assess the robustness of these attacks against defenses, revealing their resilience. Our findings highlight critical security vulnerabilities in embodied LLM systems and emphasize the urgent need for safeguarding these systems to mitigate potential risks.



## **8. ASPIRER: Bypassing System Prompts With Permutation-based Backdoors in LLMs**

cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04009v1) [paper-pdf](http://arxiv.org/pdf/2410.04009v1)

**Authors**: Lu Yan, Siyuan Cheng, Xuan Chen, Kaiyuan Zhang, Guangyu Shen, Zhuo Zhang, Xiangyu Zhang

**Abstract**: Large Language Models (LLMs) have become integral to many applications, with system prompts serving as a key mechanism to regulate model behavior and ensure ethical outputs. In this paper, we introduce a novel backdoor attack that systematically bypasses these system prompts, posing significant risks to the AI supply chain. Under normal conditions, the model adheres strictly to its system prompts. However, our backdoor allows malicious actors to circumvent these safeguards when triggered. Specifically, we explore a scenario where an LLM provider embeds a covert trigger within the base model. A downstream deployer, unaware of the hidden trigger, fine-tunes the model and offers it as a service to users. Malicious actors can purchase the trigger from the provider and use it to exploit the deployed model, disabling system prompts and achieving restricted outcomes. Our attack utilizes a permutation trigger, which activates only when its components are arranged in a precise order, making it computationally challenging to detect or reverse-engineer. We evaluate our approach on five state-of-the-art models, demonstrating that our method achieves an attack success rate (ASR) of up to 99.50% while maintaining a clean accuracy (CACC) of 98.58%, even after defensive fine-tuning. These findings highlight critical vulnerabilities in LLM deployment pipelines and underscore the need for stronger defenses.



## **9. You Know What I'm Saying -- Jailbreak Attack via Implicit Reference**

cs.CL

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03857v1) [paper-pdf](http://arxiv.org/pdf/2410.03857v1)

**Authors**: Tianyu Wu, Lingrui Mei, Ruibin Yuan, Lujun Li, Wei Xue, Yike Guo

**Abstract**: While recent advancements in large language model (LLM) alignment have enabled the effective identification of malicious objectives involving scene nesting and keyword rewriting, our study reveals that these methods remain inadequate at detecting malicious objectives expressed through context within nested harmless objectives. This study identifies a previously overlooked vulnerability, which we term Attack via Implicit Reference (AIR). AIR decomposes a malicious objective into permissible objectives and links them through implicit references within the context. This method employs multiple related harmless objectives to generate malicious content without triggering refusal responses, thereby effectively bypassing existing detection techniques.Our experiments demonstrate AIR's effectiveness across state-of-the-art LLMs, achieving an attack success rate (ASR) exceeding 90% on most models, including GPT-4o, Claude-3.5-Sonnet, and Qwen-2-72B. Notably, we observe an inverse scaling phenomenon, where larger models are more vulnerable to this attack method. These findings underscore the urgent need for defense mechanisms capable of understanding and preventing contextual attacks. Furthermore, we introduce a cross-model attack strategy that leverages less secure models to generate malicious contexts, thereby further increasing the ASR when targeting other models.Our code and jailbreak artifacts can be found at https://github.com/Lucas-TY/llm_Implicit_reference.



## **10. Detecting Machine-Generated Long-Form Content with Latent-Space Variables**

cs.CL

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03856v1) [paper-pdf](http://arxiv.org/pdf/2410.03856v1)

**Authors**: Yufei Tian, Zeyu Pan, Nanyun Peng

**Abstract**: The increasing capability of large language models (LLMs) to generate fluent long-form texts is presenting new challenges in distinguishing machine-generated outputs from human-written ones, which is crucial for ensuring authenticity and trustworthiness of expressions. Existing zero-shot detectors primarily focus on token-level distributions, which are vulnerable to real-world domain shifts, including different prompting and decoding strategies, and adversarial attacks. We propose a more robust method that incorporates abstract elements, such as event transitions, as key deciding factors to detect machine versus human texts by training a latent-space model on sequences of events or topics derived from human-written texts. In three different domains, machine-generated texts, which are originally inseparable from human texts on the token level, can be better distinguished with our latent-space model, leading to a 31% improvement over strong baselines such as DetectGPT. Our analysis further reveals that, unlike humans, modern LLMs like GPT-4 generate event triggers and their transitions differently, an inherent disparity that helps our method to robustly detect machine-generated texts.



## **11. RAFT: Realistic Attacks to Fool Text Detectors**

cs.CL

Accepted by EMNLP 2024

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03658v1) [paper-pdf](http://arxiv.org/pdf/2410.03658v1)

**Authors**: James Wang, Ran Li, Junfeng Yang, Chengzhi Mao

**Abstract**: Large language models (LLMs) have exhibited remarkable fluency across various tasks. However, their unethical applications, such as disseminating disinformation, have become a growing concern. Although recent works have proposed a number of LLM detection methods, their robustness and reliability remain unclear. In this paper, we present RAFT: a grammar error-free black-box attack against existing LLM detectors. In contrast to previous attacks for language models, our method exploits the transferability of LLM embeddings at the word-level while preserving the original text quality. We leverage an auxiliary embedding to greedily select candidate words to perturb against the target detector. Experiments reveal that our attack effectively compromises all detectors in the study across various domains by up to 99%, and are transferable across source models. Manual human evaluation studies show our attacks are realistic and indistinguishable from original human-written text. We also show that examples generated by RAFT can be used to train adversarially robust detectors. Our work shows that current LLM detectors are not adversarially robust, underscoring the urgent need for more resilient detection mechanisms.



## **12. Buckle Up: Robustifying LLMs at Every Customization Stage via Data Curation**

cs.CR

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.02220v2) [paper-pdf](http://arxiv.org/pdf/2410.02220v2)

**Authors**: Xiaoqun Liu, Jiacheng Liang, Luoxi Tang, Chenyu You, Muchao Ye, Zhaohan Xi

**Abstract**: Large language models (LLMs) are extensively adapted for downstream applications through a process known as "customization," with fine-tuning being a common method for integrating domain-specific expertise. However, recent studies have revealed a vulnerability that tuning LLMs with malicious samples can compromise their robustness and amplify harmful content, an attack known as "jailbreaking." To mitigate such attack, we propose an effective defensive framework utilizing data curation to revise commonsense texts and enhance their safety implication from the perspective of LLMs. The curated texts can mitigate jailbreaking attacks at every stage of the customization process: before customization to immunize LLMs against future jailbreak attempts, during customization to neutralize jailbreaking risks, or after customization to restore the compromised models. Since the curated data strengthens LLMs through the standard fine-tuning workflow, we do not introduce additional modules during LLM inference, thereby preserving the original customization process. Experimental results demonstrate a substantial reduction in jailbreaking effects, with up to a 100% success in generating responsible responses. Notably, our method is effective even with commonsense texts, which are often more readily available than safety-relevant data. With the every-stage defensive framework and supporting experimental performance, this work represents a significant advancement in mitigating jailbreaking risks and ensuring the secure customization of LLMs.



## **13. Jailbreaking as a Reward Misspecification Problem**

cs.LG

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2406.14393v3) [paper-pdf](http://arxiv.org/pdf/2406.14393v3)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.



## **14. AutoPenBench: Benchmarking Generative Agents for Penetration Testing**

cs.CR

Codes for the benchmark:  https://github.com/lucagioacchini/auto-pen-bench Codes for the paper  experiments: https://github.com/lucagioacchini/genai-pentest-paper

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03225v1) [paper-pdf](http://arxiv.org/pdf/2410.03225v1)

**Authors**: Luca Gioacchini, Marco Mellia, Idilio Drago, Alexander Delsanto, Giuseppe Siracusano, Roberto Bifulco

**Abstract**: Generative AI agents, software systems powered by Large Language Models (LLMs), are emerging as a promising approach to automate cybersecurity tasks. Among the others, penetration testing is a challenging field due to the task complexity and the diverse strategies to simulate cyber-attacks. Despite growing interest and initial studies in automating penetration testing with generative agents, there remains a significant gap in the form of a comprehensive and standard framework for their evaluation and development. This paper introduces AutoPenBench, an open benchmark for evaluating generative agents in automated penetration testing. We present a comprehensive framework that includes 33 tasks, each representing a vulnerable system that the agent has to attack. Tasks are of increasing difficulty levels, including in-vitro and real-world scenarios. We assess the agent performance with generic and specific milestones that allow us to compare results in a standardised manner and understand the limits of the agent under test. We show the benefits of AutoPenBench by testing two agent architectures: a fully autonomous and a semi-autonomous supporting human interaction. We compare their performance and limitations. For example, the fully autonomous agent performs unsatisfactorily achieving a 21% Success Rate (SR) across the benchmark, solving 27% of the simple tasks and only one real-world task. In contrast, the assisted agent demonstrates substantial improvements, with 64% of SR. AutoPenBench allows us also to observe how different LLMs like GPT-4o or OpenAI o1 impact the ability of the agents to complete the tasks. We believe that our benchmark fills the gap with a standard and flexible framework to compare penetration testing agents on a common ground. We hope to extend AutoPenBench along with the research community by making it available under https://github.com/lucagioacchini/auto-pen-bench.



## **15. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.02240v2) [paper-pdf](http://arxiv.org/pdf/2410.02240v2)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our code can be found at https://github.com/Pan-Zihao/SCA.



## **16. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

cs.CL

Accepted to EMNLP 2024 Findings

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2401.07867v3) [paper-pdf](http://arxiv.org/pdf/2401.07867v3)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of recent Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause evasion of automated detection in all tested languages, where homoglyph attacks are especially successful. However, some of the AO methods severely damaged the text, making it no longer readable or easily recognizable by humans (e.g., changed language, weird characters).



## **17. MoJE: Mixture of Jailbreak Experts, Naive Tabular Classifiers as Guard for Prompt Attacks**

cs.CR

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2409.17699v3) [paper-pdf](http://arxiv.org/pdf/2409.17699v3)

**Authors**: Giandomenico Cornacchia, Giulio Zizzo, Kieran Fraser, Muhammad Zaid Hameed, Ambrish Rawat, Mark Purcell

**Abstract**: The proliferation of Large Language Models (LLMs) in diverse applications underscores the pressing need for robust security measures to thwart potential jailbreak attacks. These attacks exploit vulnerabilities within LLMs, endanger data integrity and user privacy. Guardrails serve as crucial protective mechanisms against such threats, but existing models often fall short in terms of both detection accuracy, and computational efficiency. This paper advocates for the significance of jailbreak attack prevention on LLMs, and emphasises the role of input guardrails in safeguarding these models. We introduce MoJE (Mixture of Jailbreak Expert), a novel guardrail architecture designed to surpass current limitations in existing state-of-the-art guardrails. By employing simple linguistic statistical techniques, MoJE excels in detecting jailbreak attacks while maintaining minimal computational overhead during model inference. Through rigorous experimentation, MoJE demonstrates superior performance capable of detecting 90% of the attacks without compromising benign prompts, enhancing LLMs security against jailbreak attacks.



## **18. Can Watermarked LLMs be Identified by Users via Crafted Prompts?**

cs.CR

25 pages, 5 figures, 8 tables

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03168v1) [paper-pdf](http://arxiv.org/pdf/2410.03168v1)

**Authors**: Aiwei Liu, Sheng Guan, Yiming Liu, Leyi Pan, Yifei Zhang, Liancheng Fang, Lijie Wen, Philip S. Yu, Xuming Hu

**Abstract**: Text watermarking for Large Language Models (LLMs) has made significant progress in detecting LLM outputs and preventing misuse. Current watermarking techniques offer high detectability, minimal impact on text quality, and robustness to text editing. However, current researches lack investigation into the imperceptibility of watermarking techniques in LLM services. This is crucial as LLM providers may not want to disclose the presence of watermarks in real-world scenarios, as it could reduce user willingness to use the service and make watermarks more vulnerable to attacks. This work is the first to investigate the imperceptibility of watermarked LLMs. We design an identification algorithm called Water-Probe that detects watermarks through well-designed prompts to the LLM. Our key motivation is that current watermarked LLMs expose consistent biases under the same watermark key, resulting in similar differences across prompts under different watermark keys. Experiments show that almost all mainstream watermarking algorithms are easily identified with our well-designed prompts, while Water-Probe demonstrates a minimal false positive rate for non-watermarked LLMs. Finally, we propose that the key to enhancing the imperceptibility of watermarked LLMs is to increase the randomness of watermark key selection. Based on this, we introduce the Water-Bag strategy, which significantly improves watermark imperceptibility by merging multiple watermark keys.



## **19. Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs**

cs.CR

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2406.09324v2) [paper-pdf](http://arxiv.org/pdf/2406.09324v2)

**Authors**: Zhao Xu, Fan Liu, Hao Liu

**Abstract**: Although Large Language Models (LLMs) have demonstrated significant capabilities in executing complex tasks in a zero-shot manner, they are susceptible to jailbreak attacks and can be manipulated to produce harmful outputs. Recently, a growing body of research has categorized jailbreak attacks into token-level and prompt-level attacks. However, previous work primarily overlooks the diverse key factors of jailbreak attacks, with most studies concentrating on LLM vulnerabilities and lacking exploration of defense-enhanced LLMs. To address these issues, we evaluate the impact of various attack settings on LLM performance and provide a baseline benchmark for jailbreak attacks, encouraging the adoption of a standardized evaluation framework. Specifically, we evaluate the eight key factors of implementing jailbreak attacks on LLMs from both target-level and attack-level perspectives. We further conduct seven representative jailbreak attacks on six defense methods across two widely used datasets, encompassing approximately 354 experiments with about 55,000 GPU hours on A800-80G. Our experimental results highlight the need for standardized benchmarking to evaluate these attacks on defense-enhanced LLMs. Our code is available at https://github.com/usail-hkust/Bag_of_Tricks_for_LLM_Jailbreaking.



## **20. GoldCoin: Grounding Large Language Models in Privacy Laws via Contextual Integrity Theory**

cs.CL

Accepted by EMNLP 2024

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2406.11149v2) [paper-pdf](http://arxiv.org/pdf/2406.11149v2)

**Authors**: Wei Fan, Haoran Li, Zheye Deng, Weiqi Wang, Yangqiu Song

**Abstract**: Privacy issues arise prominently during the inappropriate transmission of information between entities. Existing research primarily studies privacy by exploring various privacy attacks, defenses, and evaluations within narrowly predefined patterns, while neglecting that privacy is not an isolated, context-free concept limited to traditionally sensitive data (e.g., social security numbers), but intertwined with intricate social contexts that complicate the identification and analysis of potential privacy violations. The advent of Large Language Models (LLMs) offers unprecedented opportunities for incorporating the nuanced scenarios outlined in privacy laws to tackle these complex privacy issues. However, the scarcity of open-source relevant case studies restricts the efficiency of LLMs in aligning with specific legal statutes. To address this challenge, we introduce a novel framework, GoldCoin, designed to efficiently ground LLMs in privacy laws for judicial assessing privacy violations. Our framework leverages the theory of contextual integrity as a bridge, creating numerous synthetic scenarios grounded in relevant privacy statutes (e.g., HIPAA), to assist LLMs in comprehending the complex contexts for identifying privacy risks in the real world. Extensive experimental results demonstrate that GoldCoin markedly enhances LLMs' capabilities in recognizing privacy risks across real court cases, surpassing the baselines on different judicial tasks.



## **21. Safeguard is a Double-edged Sword: Denial-of-service Attack on Large Language Models**

cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02916v1) [paper-pdf](http://arxiv.org/pdf/2410.02916v1)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern of large language models (LLMs) in their open deployment. To this end, safeguard methods aim to enforce the ethical and responsible use of LLMs through safety alignment or guardrail mechanisms. However, we found that the malicious attackers could exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a new denial-of-service (DoS) attack on LLMs. Specifically, by software or phishing attacks on user client software, attackers insert a short, seemingly innocuous adversarial prompt into to user prompt templates in configuration files; thus, this prompt appears in final user requests without visibility in the user interface and is not trivial to identify. By designing an optimization process that utilizes gradient and attention information, our attack can automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97\% of user requests on Llama Guard 3. The attack presents a new dimension of evaluating LLM safeguards focusing on false positives, fundamentally different from the classic jailbreak.



## **22. Universally Optimal Watermarking Schemes for LLMs: from Theory to Practice**

cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02890v1) [paper-pdf](http://arxiv.org/pdf/2410.02890v1)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Large Language Models (LLMs) boosts human efficiency but also poses misuse risks, with watermarking serving as a reliable method to differentiate AI-generated content from human-created text. In this work, we propose a novel theoretical framework for watermarking LLMs. Particularly, we jointly optimize both the watermarking scheme and detector to maximize detection performance, while controlling the worst-case Type-I error and distortion in the watermarked text. Within our framework, we characterize the universally minimum Type-II error, showing a fundamental trade-off between detection performance and distortion. More importantly, we identify the optimal type of detectors and watermarking schemes. Building upon our theoretical analysis, we introduce a practical, model-agnostic and computationally efficient token-level watermarking algorithm that invokes a surrogate model and the Gumbel-max trick. Empirical results on Llama-13B and Mistral-8$\times$7B demonstrate the effectiveness of our method. Furthermore, we also explore how robustness can be integrated into our theoretical framework, which provides a foundation for designing future watermarking systems with improved resilience to adversarial attacks.



## **23. Mitigating Dialogue Hallucination for Large Vision Language Models via Adversarial Instruction Tuning**

cs.CV

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2403.10492v3) [paper-pdf](http://arxiv.org/pdf/2403.10492v3)

**Authors**: Dongmin Park, Zhaofang Qian, Guangxing Han, Ser-Nam Lim

**Abstract**: Mitigating hallucinations of Large Vision Language Models,(LVLMs) is crucial to enhance their reliability for general-purpose assistants. This paper shows that such hallucinations of LVLMs can be significantly exacerbated by preceding user-system dialogues. To precisely measure this, we first present an evaluation benchmark by extending popular multi-modal benchmark datasets with prepended hallucinatory dialogues powered by our novel Adversarial Question Generator (AQG), which can automatically generate image-related yet adversarial dialogues by adopting adversarial attacks on LVLMs. On our benchmark, the zero-shot performance of state-of-the-art LVLMs drops significantly for both the VQA and Captioning tasks. Next, we further reveal this hallucination is mainly due to the prediction bias toward preceding dialogues rather than visual content. To reduce this bias, we propose Adversarial Instruction Tuning (AIT) that robustly fine-tunes LVLMs against hallucinatory dialogues. Extensive experiments show our proposed approach successfully reduces dialogue hallucination while maintaining performance.



## **24. Jailbreaking LLMs with Arabic Transliteration and Arabizi**

cs.LG

Accepted by EMNLP 2024

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2406.18725v2) [paper-pdf](http://arxiv.org/pdf/2406.18725v2)

**Authors**: Mansour Al Ghanim, Saleh Almohaimeed, Mengxin Zheng, Yan Solihin, Qian Lou

**Abstract**: This study identifies the potential vulnerabilities of Large Language Models (LLMs) to 'jailbreak' attacks, specifically focusing on the Arabic language and its various forms. While most research has concentrated on English-based prompt manipulation, our investigation broadens the scope to investigate the Arabic language. We initially tested the AdvBench benchmark in Standardized Arabic, finding that even with prompt manipulation techniques like prefix injection, it was insufficient to provoke LLMs into generating unsafe content. However, when using Arabic transliteration and chatspeak (or arabizi), we found that unsafe content could be produced on platforms like OpenAI GPT-4 and Anthropic Claude 3 Sonnet. Our findings suggest that using Arabic and its various forms could expose information that might remain hidden, potentially increasing the risk of jailbreak attacks. We hypothesize that this exposure could be due to the model's learned connection to specific words, highlighting the need for more comprehensive safety training across all language forms.



## **25. Immunization against harmful fine-tuning attacks**

cs.CL

Published in EMNLP 2024

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2402.16382v2) [paper-pdf](http://arxiv.org/pdf/2402.16382v2)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, Jan Batzner, Hassan Sajjad, Frank Rudzicz

**Abstract**: Large Language Models (LLMs) are often trained with safety guards intended to prevent harmful text generation. However, such safety training can be removed by fine-tuning the LLM on harmful datasets. While this emerging threat (harmful fine-tuning attacks) has been characterized by previous work, there is little understanding of how we should proceed in constructing and validating defenses against these attacks especially in the case where defenders would not have control of the fine-tuning process. We introduce a formal framework based on the training budget of an attacker which we call "Immunization" conditions. Using a formal characterisation of the harmful fine-tuning problem, we provide a thorough description of what a successful defense must comprise of and establish a set of guidelines on how rigorous defense research that gives us confidence should proceed.



## **26. Undesirable Memorization in Large Language Models: A Survey**

cs.CL

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02650v1) [paper-pdf](http://arxiv.org/pdf/2410.02650v1)

**Authors**: Ali Satvaty, Suzan Verberne, Fatih Turkmen

**Abstract**: While recent research increasingly showcases the remarkable capabilities of Large Language Models (LLMs), it's vital to confront their hidden pitfalls. Among these challenges, the issue of memorization stands out, posing significant ethical and legal risks. In this paper, we presents a Systematization of Knowledge (SoK) on the topic of memorization in LLMs. Memorization is the effect that a model tends to store and reproduce phrases or passages from the training data and has been shown to be the fundamental issue to various privacy and security attacks against LLMs.   We begin by providing an overview of the literature on the memorization, exploring it across five key dimensions: intentionality, degree, retrievability, abstraction, and transparency. Next, we discuss the metrics and methods used to measure memorization, followed by an analysis of the factors that contribute to memorization phenomenon. We then examine how memorization manifests itself in specific model architectures and explore strategies for mitigating these effects. We conclude our overview by identifying potential research topics for the near future: to develop methods for balancing performance and privacy in LLMs, and the analysis of memorization in specific contexts, including conversational agents, retrieval-augmented generation, multilingual language models, and diffusion language models.



## **27. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**

cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02644v1) [paper-pdf](http://arxiv.org/pdf/2410.02644v1)

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 23 different types of attack/defense methods, and 8 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, a mixed attack, and 10 corresponding defenses across 13 LLM backbones with nearly 90,000 testing cases in total. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. Our code can be found at https://github.com/agiresearch/ASB.



## **28. BadRobot: Manipulating Embodied LLMs in the Physical World**

cs.CY

38 pages, 16 figures

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2407.20242v3) [paper-pdf](http://arxiv.org/pdf/2407.20242v3)

**Authors**: Hangtao Zhang, Chenyu Zhu, Xianlong Wang, Ziqi Zhou, Changgan Yin, Minghui Li, Lulu Xue, Yichen Wang, Shengshan Hu, Aishan Liu, Peijin Guo, Leo Yu Zhang

**Abstract**: Embodied AI represents systems where AI is integrated into physical entities, enabling them to perceive and interact with their surroundings. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot. Warning: This paper contains harmful AI-generated language and aggressive actions.



## **29. Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems**

cs.MA

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02506v1) [paper-pdf](http://arxiv.org/pdf/2410.02506v1)

**Authors**: Guibin Zhang, Yanwei Yue, Zhixun Li, Sukwon Yun, Guancheng Wan, Kun Wang, Dawei Cheng, Jeffrey Xu Yu, Tianlong Chen

**Abstract**: Recent advancements in large language model (LLM)-powered agents have shown that collective intelligence can significantly outperform individual capabilities, largely attributed to the meticulously designed inter-agent communication topologies. Though impressive in performance, existing multi-agent pipelines inherently introduce substantial token overhead, as well as increased economic costs, which pose challenges for their large-scale deployments. In response to this challenge, we propose an economical, simple, and robust multi-agent communication framework, termed $\texttt{AgentPrune}$, which can seamlessly integrate into mainstream multi-agent systems and prunes redundant or even malicious communication messages. Technically, $\texttt{AgentPrune}$ is the first to identify and formally define the \textit{communication redundancy} issue present in current LLM-based multi-agent pipelines, and efficiently performs one-shot pruning on the spatial-temporal message-passing graph, yielding a token-economic and high-performing communication topology. Extensive experiments across six benchmarks demonstrate that $\texttt{AgentPrune}$ \textbf{(I)} achieves comparable results as state-of-the-art topologies at merely $\$5.6$ cost compared to their $\$43.7$, \textbf{(II)} integrates seamlessly into existing multi-agent frameworks with $28.1\%\sim72.8\%\downarrow$ token reduction, and \textbf{(III)} successfully defend against two types of agent-based adversarial attacks with $3.5\%\sim10.8\%\uparrow$ performance boost.



## **30. Demonstration Attack against In-Context Learning for Code Intelligence**

cs.CR

17 pages, 5 figures

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02841v1) [paper-pdf](http://arxiv.org/pdf/2410.02841v1)

**Authors**: Yifei Ge, Weisong Sun, Yihang Lou, Chunrong Fang, Yiran Zhang, Yiming Li, Xiaofang Zhang, Yang Liu, Zhihong Zhao, Zhenyu Chen

**Abstract**: Recent advancements in large language models (LLMs) have revolutionized code intelligence by improving programming productivity and alleviating challenges faced by software developers. To further improve the performance of LLMs on specific code intelligence tasks and reduce training costs, researchers reveal a new capability of LLMs: in-context learning (ICL). ICL allows LLMs to learn from a few demonstrations within a specific context, achieving impressive results without parameter updating. However, the rise of ICL introduces new security vulnerabilities in the code intelligence field. In this paper, we explore a novel security scenario based on the ICL paradigm, where attackers act as third-party ICL agencies and provide users with bad ICL content to mislead LLMs outputs in code intelligence tasks. Our study demonstrates the feasibility and risks of such a scenario, revealing how attackers can leverage malicious demonstrations to construct bad ICL content and induce LLMs to produce incorrect outputs, posing significant threats to system security. We propose a novel method to construct bad ICL content called DICE, which is composed of two stages: Demonstration Selection and Bad ICL Construction, constructing targeted bad ICL content based on the user query and transferable across different query inputs. Ultimately, our findings emphasize the critical importance of securing ICL mechanisms to protect code intelligence systems from adversarial manipulation.



## **31. Optimizing Adaptive Attacks against Content Watermarks for Language Models**

cs.CR

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2410.02440v1) [paper-pdf](http://arxiv.org/pdf/2410.02440v1)

**Authors**: Abdulrahman Diaa, Toluwani Aremu, Nils Lukas

**Abstract**: Large Language Models (LLMs) can be \emph{misused} to spread online spam and misinformation. Content watermarking deters misuse by hiding a message in model-generated outputs, enabling their detection using a secret watermarking key. Robustness is a core security property, stating that evading detection requires (significant) degradation of the content's quality. Many LLM watermarking methods have been proposed, but robustness is tested only against \emph{non-adaptive} attackers who lack knowledge of the watermarking method and can find only suboptimal attacks. We formulate the robustness of LLM watermarking as an objective function and propose preference-based optimization to tune \emph{adaptive} attacks against the specific watermarking method. Our evaluation shows that (i) adaptive attacks substantially outperform non-adaptive baselines. (ii) Even in a non-adaptive setting, adaptive attacks optimized against a few known watermarks remain highly effective when tested against other unseen watermarks, and (iii) optimization-based attacks are practical and require less than seven GPU hours. Our findings underscore the need to test robustness against adaptive attackers.



## **32. Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models**

cs.CR

10 pages, 5 figures

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.02298v2) [paper-pdf](http://arxiv.org/pdf/2410.02298v2)

**Authors**: Guobin Shen, Dongcheng Zhao, Yiting Dong, Xiang He, Yi Zeng

**Abstract**: As large language models (LLMs) become integral to various applications, ensuring both their safety and utility is paramount. Jailbreak attacks, which manipulate LLMs into generating harmful content, pose significant challenges to this balance. Existing defenses, such as prompt engineering and safety fine-tuning, often introduce computational overhead, increase inference latency, and lack runtime flexibility. Moreover, overly restrictive safety measures can degrade model utility by causing refusals of benign queries. In this paper, we introduce Jailbreak Antidote, a method that enables real-time adjustment of LLM safety preferences by manipulating a sparse subset of the model's internal states during inference. By shifting the model's hidden representations along a safety direction with varying strengths, we achieve flexible control over the safety-utility balance without additional token overhead or inference delays. Our analysis reveals that safety-related information in LLMs is sparsely distributed; adjusting approximately 5% of the internal state is as effective as modifying the entire state. Extensive experiments on nine LLMs (ranging from 2 billion to 72 billion parameters), evaluated against ten jailbreak attack methods and compared with six defense strategies, validate the effectiveness and efficiency of our approach. By directly manipulating internal states during reasoning, Jailbreak Antidote offers a lightweight, scalable solution that enhances LLM safety while preserving utility, opening new possibilities for real-time safety mechanisms in widely-deployed AI systems.



## **33. PathSeeker: Exploring LLM Security Vulnerabilities with a Reinforcement Learning-Based Jailbreak Approach**

cs.CR

update the abstract and cite a new related work

**SubmitDate**: 2024-10-03    [abs](http://arxiv.org/abs/2409.14177v2) [paper-pdf](http://arxiv.org/pdf/2409.14177v2)

**Authors**: Zhihao Lin, Wei Ma, Mingyi Zhou, Yanjie Zhao, Haoyu Wang, Yang Liu, Jun Wang, Li Li

**Abstract**: In recent years, Large Language Models (LLMs) have gained widespread use, raising concerns about their security. Traditional jailbreak attacks, which often rely on the model internal information or have limitations when exploring the unsafe behavior of the victim model, limiting their reducing their general applicability. In this paper, we introduce PathSeeker, a novel black-box jailbreak method, which is inspired by the game of rats escaping a maze. We think that each LLM has its unique "security maze", and attackers attempt to find the exit learning from the received feedback and their accumulated experience to compromise the target LLM's security defences. Our approach leverages multi-agent reinforcement learning, where smaller models collaborate to guide the main LLM in performing mutation operations to achieve the attack objectives. By progressively modifying inputs based on the model's feedback, our system induces richer, harmful responses. During our manual attempts to perform jailbreak attacks, we found that the vocabulary of the response of the target model gradually became richer and eventually produced harmful responses. Based on the observation, we also introduce a reward mechanism that exploits the expansion of vocabulary richness in LLM responses to weaken security constraints. Our method outperforms five state-of-the-art attack techniques when tested across 13 commercial and open-source LLMs, achieving high attack success rates, especially in strongly aligned commercial models like GPT-4o-mini, Claude-3.5, and GLM-4-air with strong safety alignment. This study aims to improve the understanding of LLM security vulnerabilities and we hope that this sturdy can contribute to the development of more robust defenses.



## **34. DomainLynx: Leveraging Large Language Models for Enhanced Domain Squatting Detection**

cs.CR

Accepted for publication at IEEE CCNC 2025

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.02095v1) [paper-pdf](http://arxiv.org/pdf/2410.02095v1)

**Authors**: Daiki Chiba, Hiroki Nakano, Takashi Koide

**Abstract**: Domain squatting poses a significant threat to Internet security, with attackers employing increasingly sophisticated techniques. This study introduces DomainLynx, an innovative compound AI system leveraging Large Language Models (LLMs) for enhanced domain squatting detection. Unlike existing methods focusing on predefined patterns for top-ranked domains, DomainLynx excels in identifying novel squatting techniques and protecting less prominent brands. The system's architecture integrates advanced data processing, intelligent domain pairing, and LLM-powered threat assessment. Crucially, DomainLynx incorporates specialized components that mitigate LLM hallucinations, ensuring reliable and context-aware detection. This approach enables efficient analysis of vast security data from diverse sources, including Certificate Transparency logs, Passive DNS records, and zone files. Evaluated on a curated dataset of 1,649 squatting domains, DomainLynx achieved 94.7\% accuracy using Llama-3-70B. In a month-long real-world test, it detected 34,359 squatting domains from 2.09 million new domains, outperforming baseline methods by 2.5 times. This research advances Internet security by providing a versatile, accurate, and adaptable tool for combating evolving domain squatting threats. DomainLynx's approach paves the way for more robust, AI-driven cybersecurity solutions, enhancing protection for a broader range of online entities and contributing to a safer digital ecosystem.



## **35. Precision Knowledge Editing: Enhancing Safety in Large Language Models**

cs.CL

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.03772v1) [paper-pdf](http://arxiv.org/pdf/2410.03772v1)

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Yasuhiro Yoshida, Victor Bian

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but they also pose risks related to the generation of toxic or harmful content. This work introduces Precision Knowledge Editing (PKE), an advanced technique that builds upon existing knowledge editing methods to more effectively identify and modify toxic parameter regions within LLMs. By leveraging neuron weight tracking and activation pathway tracing, PKE achieves finer granularity in toxic content management compared to previous methods like Detoxifying Instance Neuron Modification (DINM). Our experiments demonstrate that PKE significantly reduces the attack success rate (ASR) across various models, including Llama2-7b and Llama-3-8b-instruct, while maintaining overall model performance. Additionally, we also compared the performance of some closed-source models (gpt-4-0613 and Claude 3 Sonnet) in our experiments, and found that models adjusted using our method far outperformed the closed-source models in terms of safety. This research contributes to the ongoing efforts to make LLMs safer and more reliable for real-world applications.



## **36. TuBA: Cross-Lingual Transferability of Backdoor Attacks in LLMs with Instruction Tuning**

cs.CL

work in progress

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2404.19597v2) [paper-pdf](http://arxiv.org/pdf/2404.19597v2)

**Authors**: Xuanli He, Jun Wang, Qiongkai Xu, Pasquale Minervini, Pontus Stenetorp, Benjamin I. P. Rubinstein, Trevor Cohn

**Abstract**: The implications of backdoor attacks on English-centric large language models (LLMs) have been widely examined - such attacks can be achieved by embedding malicious behaviors during training and activated under specific conditions that trigger malicious outputs. Despite the increasing support for multilingual capabilities in open-source and proprietary LLMs, the impact of backdoor attacks on these systems remains largely under-explored. Our research focuses on cross-lingual backdoor attacks against multilingual LLMs, particularly investigating how poisoning the instruction-tuning data for one or two languages can affect the outputs for languages whose instruction-tuning data were not poisoned. Despite its simplicity, our empirical analysis reveals that our method exhibits remarkable efficacy in models like mT5 and GPT-4o, with high attack success rates, surpassing 90% in more than 7 out of 12 languages across various scenarios. Our findings also indicate that more powerful models show increased susceptibility to transferable cross-lingual backdoor attacks, which also applies to LLMs predominantly pre-trained on English data, such as Llama2, Llama3, and Gemma. Moreover, our experiments demonstrate 1) High Transferability: the backdoor mechanism operates successfully in cross-lingual response scenarios across 26 languages, achieving an average attack success rate of 99%, and 2) Robustness: the proposed attack remains effective even after defenses are applied. These findings expose critical security vulnerabilities in multilingual LLMs and highlight the urgent need for more robust, targeted defense strategies to address the unique challenges posed by cross-lingual backdoor transfer.



## **37. Automated Red Teaming with GOAT: the Generative Offensive Agent Tester**

cs.LG

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01606v1) [paper-pdf](http://arxiv.org/pdf/2410.01606v1)

**Authors**: Maya Pavlova, Erik Brinkman, Krithika Iyer, Vitor Albiero, Joanna Bitton, Hailey Nguyen, Joe Li, Cristian Canton Ferrer, Ivan Evtimov, Aaron Grattafiori

**Abstract**: Red teaming assesses how large language models (LLMs) can produce content that violates norms, policies, and rules set during their safety training. However, most existing automated methods in the literature are not representative of the way humans tend to interact with AI models. Common users of AI models may not have advanced knowledge of adversarial machine learning methods or access to model internals, and they do not spend a lot of time crafting a single highly effective adversarial prompt. Instead, they are likely to make use of techniques commonly shared online and exploit the multiturn conversational nature of LLMs. While manual testing addresses this gap, it is an inefficient and often expensive process. To address these limitations, we introduce the Generative Offensive Agent Tester (GOAT), an automated agentic red teaming system that simulates plain language adversarial conversations while leveraging multiple adversarial prompting techniques to identify vulnerabilities in LLMs. We instantiate GOAT with 7 red teaming attacks by prompting a general-purpose model in a way that encourages reasoning through the choices of methods available, the current target model's response, and the next steps. Our approach is designed to be extensible and efficient, allowing human testers to focus on exploring new areas of risk while automation covers the scaled adversarial stress-testing of known risk territory. We present the design and evaluation of GOAT, demonstrating its effectiveness in identifying vulnerabilities in state-of-the-art LLMs, with an ASR@10 of 97% against Llama 3.1 and 88% against GPT-4 on the JailbreakBench dataset.



## **38. Leveraging the Context through Multi-Round Interactions for Jailbreaking Attacks**

cs.LG

29 pages

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2402.09177v2) [paper-pdf](http://arxiv.org/pdf/2402.09177v2)

**Authors**: Yixin Cheng, Markos Georgopoulos, Volkan Cevher, Grigorios G. Chrysos

**Abstract**: Large Language Models (LLMs) are susceptible to Jailbreaking attacks, which aim to extract harmful information by subtly modifying the attack query. As defense mechanisms evolve, directly obtaining harmful information becomes increasingly challenging for Jailbreaking attacks. In this work, inspired from Chomsky's transformational-generative grammar theory and human practices of indirect context to elicit harmful information, we focus on a new attack form, called Contextual Interaction Attack. We contend that the prior context\u2014the information preceding the attack query\u2014plays a pivotal role in enabling strong Jailbreaking attacks. Specifically, we propose a first multi-turn approach that leverages benign preliminary questions to interact with the LLM. Due to the autoregressive nature of LLMs, which use previous conversation rounds as context during generation, we guide the model's question-response pair to construct a context that is semantically aligned with the attack query to execute the attack. We conduct experiments on seven different LLMs and demonstrate the efficacy of this attack, which is black-box and can also transfer across LLMs. We believe this can lead to further developments and understanding of security in LLMs.



## **39. Backdooring Vision-Language Models with Out-Of-Distribution Data**

cs.CV

**SubmitDate**: 2024-10-02    [abs](http://arxiv.org/abs/2410.01264v1) [paper-pdf](http://arxiv.org/pdf/2410.01264v1)

**Authors**: Weimin Lyu, Jiachen Yao, Saumya Gupta, Lu Pang, Tao Sun, Lingjie Yi, Lijie Hu, Haibin Ling, Chao Chen

**Abstract**: The emergence of Vision-Language Models (VLMs) represents a significant advancement in integrating computer vision with Large Language Models (LLMs) to generate detailed text descriptions from visual inputs. Despite their growing importance, the security of VLMs, particularly against backdoor attacks, is under explored. Moreover, prior works often assume attackers have access to the original training data, which is often unrealistic. In this paper, we address a more practical and challenging scenario where attackers must rely solely on Out-Of-Distribution (OOD) data. We introduce VLOOD (Backdooring Vision-Language Models with Out-of-Distribution Data), a novel approach with two key contributions: (1) demonstrating backdoor attacks on VLMs in complex image-to-text tasks while minimizing degradation of the original semantics under poisoned inputs, and (2) proposing innovative techniques for backdoor injection without requiring any access to the original training data. Our evaluation on image captioning and visual question answering (VQA) tasks confirms the effectiveness of VLOOD, revealing a critical security vulnerability in VLMs and laying the foundation for future research on securing multimodal models against sophisticated threats.



## **40. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

cs.AI

**SubmitDate**: 2024-10-01    [abs](http://arxiv.org/abs/2407.00075v2) [paper-pdf](http://arxiv.org/pdf/2407.00075v2)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We model rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form ``if $P$ and $Q$, then $R$'' for some propositions $P$, $Q$, and $R$. We prove that although LLMs can faithfully follow such rules, maliciously crafted prompts can mislead even idealized, theoretically constructed models. Empirically, we find that the reasoning behavior of LLMs aligns with that of our theoretical constructions, and popular attack algorithms find adversarial prompts with characteristics predicted by our theory. Our logic-based framework provides a novel perspective for mechanistically understanding the behavior of LLMs in rule-based settings such as jailbreak attacks.



## **41. Backdoor Attacks for LLMs with Weak-To-Strong Knowledge Distillation**

cs.CR

**SubmitDate**: 2024-10-01    [abs](http://arxiv.org/abs/2409.17946v2) [paper-pdf](http://arxiv.org/pdf/2409.17946v2)

**Authors**: Shuai Zhao, Leilei Gan, Zhongliang Guo, Xiaobao Wu, Luwei Xiao, Xiaoyu Xu, Cong-Duy Nguyen, Luu Anh Tuan

**Abstract**: Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning. However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from weak to strong based on feature alignment-enhanced knowledge distillation (W2SAttack). Specifically, we poison small-scale language models through full-parameter fine-tuning to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through feature alignment-enhanced knowledge distillation, which employs PEFT. Theoretical analysis reveals that W2SAttack has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of W2SAttack on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.



## **42. Universal Vulnerabilities in Large Language Models: Backdoor Attacks for In-context Learning**

cs.CL

**SubmitDate**: 2024-10-01    [abs](http://arxiv.org/abs/2401.05949v5) [paper-pdf](http://arxiv.org/pdf/2401.05949v5)

**Authors**: Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Fengjun Pan, Jinming Wen

**Abstract**: In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we design a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning demonstration prompts, which can make models behave in alignment with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 180B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models.



## **43. Privacy Evaluation Benchmarks for NLP Models**

cs.CL

Findings of EMNLP 2024

**SubmitDate**: 2024-10-01    [abs](http://arxiv.org/abs/2409.15868v3) [paper-pdf](http://arxiv.org/pdf/2409.15868v3)

**Authors**: Wei Huang, Yinggui Wang, Cen Chen

**Abstract**: By inducing privacy attacks on NLP models, attackers can obtain sensitive information such as training data and model parameters, etc. Although researchers have studied, in-depth, several kinds of attacks in NLP models, they are non-systematic analyses. It lacks a comprehensive understanding of the impact caused by the attacks. For example, we must consider which scenarios can apply to which attacks, what the common factors are that affect the performance of different attacks, the nature of the relationships between different attacks, and the influence of various datasets and models on the effectiveness of the attacks, etc. Therefore, we need a benchmark to holistically assess the privacy risks faced by NLP models. In this paper, we present a privacy attack and defense evaluation benchmark in the field of NLP, which includes the conventional/small models and large language models (LLMs). This benchmark supports a variety of models, datasets, and protocols, along with standardized modules for comprehensive evaluation of attacks and defense strategies. Based on the above framework, we present a study on the association between auxiliary data from different domains and the strength of privacy attacks. And we provide an improved attack method in this scenario with the help of Knowledge Distillation (KD). Furthermore, we propose a chained framework for privacy attacks. Allowing a practitioner to chain multiple attacks to achieve a higher-level attack objective. Based on this, we provide some defense and enhanced attack strategies. The code for reproducing the results can be found at https://github.com/user2311717757/nlp_doctor.



## **44. Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2409.18169v2) [paper-pdf](http://arxiv.org/pdf/2409.18169v2)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent research demonstrates that the nascent fine-tuning-as-a-service business model exposes serious safety concerns -- fine-tuning over a few harmful data uploaded by the users can compromise the safety alignment of the model. The attack, known as harmful fine-tuning, has raised a broad research interest among the community. However, as the attack is still new, \textbf{we observe from our miserable submission experience that there are general misunderstandings within the research community.} We in this paper aim to clear some common concerns for the attack setting, and formally establish the research problem. Specifically, we first present the threat model of the problem, and introduce the harmful fine-tuning attack and its variants. Then we systematically survey the existing literature on attacks/defenses/mechanical analysis of the problem. Finally, we outline future research directions that might contribute to the development of the field. Additionally, we present a list of questions of interest, which might be useful to refer to when reviewers in the peer review process question the realism of the experiment/attack/defense setting. A curated list of relevant papers is maintained and made accessible at: \url{https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers}.



## **45. Distract Large Language Models for Automatic Jailbreak Attack**

cs.CR

EMNLP 2024

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2403.08424v2) [paper-pdf](http://arxiv.org/pdf/2403.08424v2)

**Authors**: Zeguan Xiao, Yan Yang, Guanhua Chen, Yun Chen

**Abstract**: Extensive efforts have been made before the public release of Large language models (LLMs) to align their behaviors with human values. However, even meticulously aligned LLMs remain vulnerable to malicious manipulations such as jailbreaking, leading to unintended behaviors. In this work, we propose a novel black-box jailbreak framework for automated red teaming of LLMs. We designed malicious content concealing and memory reframing with an iterative optimization algorithm to jailbreak LLMs, motivated by the research about the distractibility and over-confidence phenomenon of LLMs. Extensive experiments of jailbreaking both open-source and proprietary LLMs demonstrate the superiority of our framework in terms of effectiveness, scalability and transferability. We also evaluate the effectiveness of existing jailbreak defense methods against our attack and highlight the crucial need to develop more effective and practical defense strategies.



## **46. ModelShield: Adaptive and Robust Watermark against Model Extraction Attack**

cs.CR

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2405.02365v3) [paper-pdf](http://arxiv.org/pdf/2405.02365v3)

**Authors**: Kaiyi Pang, Tao Qi, Chuhan Wu, Minhao Bai, Minghu Jiang, Yongfeng Huang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.



## **47. Privacy in Large Language Models: Attacks, Defenses and Future Directions**

cs.CL

We upload the survey to cover more recent papers and inlcude privacy  resaearch on multi-modality

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2310.10383v2) [paper-pdf](http://arxiv.org/pdf/2310.10383v2)

**Authors**: Haoran Li, Yulin Chen, Jinglong Luo, Jiecong Wang, Hao Peng, Yan Kang, Xiaojin Zhang, Qi Hu, Chunkit Chan, Zenglin Xu, Bryan Hooi, Yangqiu Song

**Abstract**: The advancement of large language models (LLMs) has significantly enhanced the ability to effectively tackle various downstream NLP tasks and unify these tasks into generative pipelines. On the one hand, powerful language models, trained on massive textual data, have brought unparalleled accessibility and usability for both models and users. On the other hand, unrestricted access to these models can also introduce potential malicious and unintentional privacy risks. Despite ongoing efforts to address the safety and privacy concerns associated with LLMs, the problem remains unresolved. In this paper, we provide a comprehensive analysis of the current privacy attacks targeting LLMs and categorize them according to the adversary's assumed capabilities to shed light on the potential vulnerabilities present in LLMs. Then, we present a detailed overview of prominent defense strategies that have been developed to counter these privacy attacks. Beyond existing works, we identify upcoming privacy concerns as LLMs evolve. Lastly, we point out several potential avenues for future exploration.



## **48. Robust LLM safeguarding via refusal feature adversarial training**

cs.LG

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2409.20089v1) [paper-pdf](http://arxiv.org/pdf/2409.20089v1)

**Authors**: Lei Yu, Virginie Do, Karen Hambardzumyan, Nicola Cancedda

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can elicit harmful responses. Defending against such attacks remains challenging due to the opacity of jailbreaking mechanisms and the high computational cost of training LLMs robustly. We demonstrate that adversarial attacks share a universal mechanism for circumventing LLM safeguards that works by ablating a dimension in the residual stream embedding space called the refusal feature. We further show that the operation of refusal feature ablation (RFA) approximates the worst-case perturbation of offsetting model safety. Based on these findings, we propose Refusal Feature Adversarial Training (ReFAT), a novel algorithm that efficiently performs LLM adversarial training by simulating the effect of input-level attacks via RFA. Experiment results show that ReFAT significantly improves the robustness of three popular LLMs against a wide range of adversarial attacks, with considerably less computational overhead compared to existing adversarial training methods.



## **49. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

cs.CR

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2409.20002v1) [paper-pdf](http://arxiv.org/pdf/2409.20002v1)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.



## **50. Mitigating Backdoor Threats to Large Language Models: Advancement and Challenges**

cs.CR

The 60th Annual Allerton Conference (Invited Paper). The arXiv  version is a pre-IEEE Press publication version

**SubmitDate**: 2024-09-30    [abs](http://arxiv.org/abs/2409.19993v1) [paper-pdf](http://arxiv.org/pdf/2409.19993v1)

**Authors**: Qin Liu, Wenjie Mo, Terry Tong, Jiashu Xu, Fei Wang, Chaowei Xiao, Muhao Chen

**Abstract**: The advancement of Large Language Models (LLMs) has significantly impacted various domains, including Web search, healthcare, and software development. However, as these models scale, they become more vulnerable to cybersecurity risks, particularly backdoor attacks. By exploiting the potent memorization capacity of LLMs, adversaries can easily inject backdoors into LLMs by manipulating a small portion of training data, leading to malicious behaviors in downstream applications whenever the hidden backdoor is activated by the pre-defined triggers. Moreover, emerging learning paradigms like instruction tuning and reinforcement learning from human feedback (RLHF) exacerbate these risks as they rely heavily on crowdsourced data and human feedback, which are not fully controlled. In this paper, we present a comprehensive survey of emerging backdoor threats to LLMs that appear during LLM development or inference, and cover recent advancement in both defense and detection strategies for mitigating backdoor threats to LLMs. We also outline key challenges in addressing these threats, highlighting areas for future research.



