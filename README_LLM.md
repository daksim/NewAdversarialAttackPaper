# Latest Large Language Model Attack Papers
**update at 2024-06-14 16:18:54**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space**

cs.CL

18 pages, 3 figures

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09325v1) [paper-pdf](http://arxiv.org/pdf/2406.09325v1)

**Authors**: Tomer Ashuach, Martin Tutek, Yonatan Belinkov

**Abstract**: Large language models (LLMs) risk inadvertently memorizing and divulging sensitive or personally identifiable information (PII) seen in training data, causing privacy concerns. Current approaches to address this issue involve costly dataset scrubbing, or model filtering through unlearning and model editing, which can be bypassed through extraction attacks. We propose REVS, a novel model editing method for unlearning sensitive information from LLMs. REVS identifies and modifies a small subset of neurons relevant for each piece of sensitive information. By projecting these neurons to the vocabulary space (unembedding), we pinpoint the components driving its generation. We then compute a model edit based on the pseudo-inverse of the unembedding matrix, and apply it to de-promote generation of the targeted sensitive data. To adequately evaluate our method on truly sensitive information, we curate two datasets: an email dataset inherently memorized by GPT-J, and a synthetic social security number dataset that we tune the model to memorize. Compared to other state-of-the-art model editing methods, REVS demonstrates superior performance in both eliminating sensitive information and robustness to extraction attacks, while retaining integrity of the underlying model. The code and a demo notebook are available at https://technion-cs-nlp.github.io/REVS.



## **2. Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs**

cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09324v1) [paper-pdf](http://arxiv.org/pdf/2406.09324v1)

**Authors**: Zhao Xu, Fan Liu, Hao Liu

**Abstract**: Although Large Language Models (LLMs) have demonstrated significant capabilities in executing complex tasks in a zero-shot manner, they are susceptible to jailbreak attacks and can be manipulated to produce harmful outputs. Recently, a growing body of research has categorized jailbreak attacks into token-level and prompt-level attacks. However, previous work primarily overlooks the diverse key factors of jailbreak attacks, with most studies concentrating on LLM vulnerabilities and lacking exploration of defense-enhanced LLMs. To address these issues, we evaluate the impact of various attack settings on LLM performance and provide a baseline benchmark for jailbreak attacks, encouraging the adoption of a standardized evaluation framework. Specifically, we evaluate the eight key factors of implementing jailbreak attacks on LLMs from both target-level and attack-level perspectives. We further conduct seven representative jailbreak attacks on six defense methods across two widely used datasets, encompassing approximately 320 experiments with about 50,000 GPU hours on A800-80G. Our experimental results highlight the need for standardized benchmarking to evaluate these attacks on defense-enhanced LLMs. Our code is available at https://github.com/usail-hkust/Bag_of_Tricks_for_LLM_Jailbreaking.



## **3. JailbreakEval: An Integrated Toolkit for Evaluating Jailbreak Attempts Against Large Language Models**

cs.CR

Our code is available at https://github.com/ThuCCSLab/JailbreakEval

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.09321v1) [paper-pdf](http://arxiv.org/pdf/2406.09321v1)

**Authors**: Delong Ran, Jinyuan Liu, Yichen Gong, Jingyi Zheng, Xinlei He, Tianshuo Cong, Anyu Wang

**Abstract**: Jailbreak attacks aim to induce Large Language Models (LLMs) to generate harmful responses for forbidden instructions, presenting severe misuse threats to LLMs. Up to now, research into jailbreak attacks and defenses is emerging, however, there is (surprisingly) no consensus on how to evaluate whether a jailbreak attempt is successful. In other words, the methods to assess the harmfulness of an LLM's response are varied, such as manual annotation or prompting GPT-4 in specific ways. Each approach has its own set of strengths and weaknesses, impacting their alignment with human values, as well as the time and financial cost. This diversity in evaluation presents challenges for researchers in choosing suitable evaluation methods and conducting fair comparisons across different jailbreak attacks and defenses. In this paper, we conduct a comprehensive analysis of jailbreak evaluation methodologies, drawing from nearly ninety jailbreak research released between May 2023 and April 2024. Our study introduces a systematic taxonomy of jailbreak evaluators, offering in-depth insights into their strengths and weaknesses, along with the current status of their adaptation. Moreover, to facilitate subsequent research, we propose JailbreakEval, a user-friendly toolkit focusing on the evaluation of jailbreak attempts. It includes various well-known evaluators out-of-the-box, so that users can obtain evaluation results with only a single command. JailbreakEval also allows users to customize their own evaluation workflow in a unified framework with the ease of development and comparison. In summary, we regard JailbreakEval to be a catalyst that simplifies the evaluation process in jailbreak research and fosters an inclusive standard for jailbreak evaluation within the community.



## **4. A Survey of Backdoor Attacks and Defenses on Large Language Models: Implications for Security Measures**

cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.06852v2) [paper-pdf](http://arxiv.org/pdf/2406.06852v2)

**Authors**: Shuai Zhao, Meihuizi Jia, Zhongliang Guo, Leilei Gan, Jie Fu, Yichao Feng, Fengjun Pan, Luu Anh Tuan

**Abstract**: The large language models (LLMs), which bridge the gap between human language understanding and complex problem-solving, achieve state-of-the-art performance on several NLP tasks, particularly in few-shot and zero-shot settings. Despite the demonstrable efficacy of LMMs, due to constraints on computational resources, users have to engage with open-source language models or outsource the entire training process to third-party platforms. However, research has demonstrated that language models are susceptible to potential security vulnerabilities, particularly in backdoor attacks. Backdoor attacks are designed to introduce targeted vulnerabilities into language models by poisoning training samples or model weights, allowing attackers to manipulate model responses through malicious triggers. While existing surveys on backdoor attacks provide a comprehensive overview, they lack an in-depth examination of backdoor attacks specifically targeting LLMs. To bridge this gap and grasp the latest trends in the field, this paper presents a novel perspective on backdoor attacks for LLMs by focusing on fine-tuning methods. Specifically, we systematically classify backdoor attacks into three categories: full-parameter fine-tuning, parameter-efficient fine-tuning, and attacks without fine-tuning. Based on insights from a substantial review, we also discuss crucial issues for future research on backdoor attacks, such as further exploring attack algorithms that do not require fine-tuning, or developing more covert attack algorithms.



## **5. StructuralSleight: Automated Jailbreak Attacks on Large Language Models Utilizing Uncommon Text-Encoded Structure**

cs.CL

12 pages, 4 figures

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.08754v1) [paper-pdf](http://arxiv.org/pdf/2406.08754v1)

**Authors**: Bangxin Li, Hengrui Xing, Chao Huang, Jin Qian, Huangqing Xiao, Linfeng Feng, Cong Tian

**Abstract**: Large Language Models (LLMs) are widely used in natural language processing but face the risk of jailbreak attacks that maliciously induce them to generate harmful content. Existing jailbreak attacks, including character-level and context-level attacks, mainly focus on the prompt of the plain text without specifically exploring the significant influence of its structure. In this paper, we focus on studying how prompt structure contributes to the jailbreak attack. We introduce a novel structure-level attack method based on tail structures that are rarely used during LLM training, which we refer to as Uncommon Text-Encoded Structure (UTES). We extensively study 12 UTESs templates and 6 obfuscation methods to build an effective automated jailbreak tool named StructuralSleight that contains three escalating attack strategies: Structural Attack, Structural and Character/Context Obfuscation Attack, and Fully Obfuscated Structural Attack. Extensive experiments on existing LLMs show that StructuralSleight significantly outperforms baseline methods. In particular, the attack success rate reaches 94.62\% on GPT-4o, which has not been addressed by state-of-the-art techniques.



## **6. Ranking Manipulation for Conversational Search Engines**

cs.CL

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.03589v2) [paper-pdf](http://arxiv.org/pdf/2406.03589v2)

**Authors**: Samuel Pfrommer, Yatong Bai, Tanmay Gautam, Somayeh Sojoudi

**Abstract**: Major search engine providers are rapidly incorporating Large Language Model (LLM)-generated content in response to user queries. These conversational search engines operate by loading retrieved website text into the LLM context for summarization and interpretation. Recent research demonstrates that LLMs are highly vulnerable to jailbreaking and prompt injection attacks, which disrupt the safety and quality goals of LLMs using adversarial strings. This work investigates the impact of prompt injections on the ranking order of sources referenced by conversational search engines. To this end, we introduce a focused dataset of real-world consumer product websites and formalize conversational search ranking as an adversarial problem. Experimentally, we analyze conversational search rankings in the absence of adversarial injections and show that different LLMs vary significantly in prioritizing product name, document content, and context position. We then present a tree-of-attacks-based jailbreaking technique which reliably promotes low-ranked products. Importantly, these attacks transfer effectively to state-of-the-art conversational search engines such as perplexity.ai. Given the strong financial incentive for website owners to boost their search ranking, we argue that our problem formulation is of critical importance for future robustness work.



## **7. RL-JACK: Reinforcement Learning-powered Black-box Jailbreaking Attack against LLMs**

cs.CR

**SubmitDate**: 2024-06-13    [abs](http://arxiv.org/abs/2406.08725v1) [paper-pdf](http://arxiv.org/pdf/2406.08725v1)

**Authors**: Xuan Chen, Yuzhou Nie, Lu Yan, Yunshu Mao, Wenbo Guo, Xiangyu Zhang

**Abstract**: Modern large language model (LLM) developers typically conduct a safety alignment to prevent an LLM from generating unethical or harmful content. Recent studies have discovered that the safety alignment of LLMs can be bypassed by jailbreaking prompts. These prompts are designed to create specific conversation scenarios with a harmful question embedded. Querying an LLM with such prompts can mislead the model into responding to the harmful question. The stochastic and random nature of existing genetic methods largely limits the effectiveness and efficiency of state-of-the-art (SOTA) jailbreaking attacks. In this paper, we propose RL-JACK, a novel black-box jailbreaking attack powered by deep reinforcement learning (DRL). We formulate the generation of jailbreaking prompts as a search problem and design a novel RL approach to solve it. Our method includes a series of customized designs to enhance the RL agent's learning efficiency in the jailbreaking context. Notably, we devise an LLM-facilitated action space that enables diverse action variations while constraining the overall search space. We propose a novel reward function that provides meaningful dense rewards for the agent toward achieving successful jailbreaking. Through extensive evaluations, we demonstrate that RL-JACK is overall much more effective than existing jailbreaking attacks against six SOTA LLMs, including large open-source models and commercial models. We also show the RL-JACK's resiliency against three SOTA defenses and its transferability across different models. Finally, we validate the insensitivity of RL-JACK to the variations in key hyper-parameters.



## **8. Adversarial Evasion Attack Efficiency against Large Language Models**

cs.CL

9 pages, 1 table, 2 figures, DCAI 2024 conference

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.08050v1) [paper-pdf](http://arxiv.org/pdf/2406.08050v1)

**Authors**: João Vitorino, Eva Maia, Isabel Praça

**Abstract**: Large Language Models (LLMs) are valuable for text classification, but their vulnerabilities must not be disregarded. They lack robustness against adversarial examples, so it is pertinent to understand the impacts of different types of perturbations, and assess if those attacks could be replicated by common users with a small amount of perturbations and a small number of queries to a deployed LLM. This work presents an analysis of the effectiveness, efficiency, and practicality of three different types of adversarial attacks against five different LLMs in a sentiment classification task. The obtained results demonstrated the very distinct impacts of the word-level and character-level attacks. The word attacks were more effective, but the character and more constrained attacks were more practical and required a reduced number of perturbations and queries. These differences need to be considered during the development of adversarial defense strategies to train more robust LLMs for intelligent text classification applications.



## **9. Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization**

cs.CL

ACL 2024 Main Conference

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2311.09096v2) [paper-pdf](http://arxiv.org/pdf/2311.09096v2)

**Authors**: Zhexin Zhang, Junxiao Yang, Pei Ke, Fei Mi, Hongning Wang, Minlie Huang

**Abstract**: While significant attention has been dedicated to exploiting weaknesses in LLMs through jailbreaking attacks, there remains a paucity of effort in defending against these attacks. We point out a pivotal factor contributing to the success of jailbreaks: the intrinsic conflict between the goals of being helpful and ensuring safety. Accordingly, we propose to integrate goal prioritization at both training and inference stages to counteract. Implementing goal prioritization during inference substantially diminishes the Attack Success Rate (ASR) of jailbreaking from 66.4% to 3.6% for ChatGPT. And integrating goal prioritization into model training reduces the ASR from 71.0% to 6.6% for Llama2-13B. Remarkably, even in scenarios where no jailbreaking samples are included during training, our approach slashes the ASR by half. Additionally, our findings reveal that while stronger LLMs face greater safety risks, they also possess a greater capacity to be steered towards defending against such attacks, both because of their stronger ability in instruction following. Our work thus contributes to the comprehension of jailbreaking attacks and defenses, and sheds light on the relationship between LLMs' capability and safety. Our code is available at \url{https://github.com/thu-coai/JailbreakDefense_GoalPriority}.



## **10. Unique Security and Privacy Threats of Large Language Model: A Comprehensive Survey**

cs.CR

38 pages, 17 figures, 4 tables

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.07973v1) [paper-pdf](http://arxiv.org/pdf/2406.07973v1)

**Authors**: Shang Wang, Tianqing Zhu, Bo Liu, Ding Ming, Xu Guo, Dayong Ye, Wanlei Zhou

**Abstract**: With the rapid development of artificial intelligence, large language models (LLMs) have made remarkable progress in natural language processing. These models are trained on large amounts of data to demonstrate powerful language understanding and generation capabilities for various applications, from machine translation and chatbots to agents. However, LLMs have exposed a variety of privacy and security issues during their life cycle, which have become the focus of academic and industrial attention. Moreover, these risks LLMs face are pretty different from previous traditional language models. Since current surveys lack a clear taxonomy of unique threat models based on diverse scenarios, we highlight unique privacy and security issues based on five scenarios: pre-training, fine-tuning, RAG system, deploying, and LLM-based agent. Concerning the characteristics of each risk, this survey provides potential threats and countermeasures. The research on attack and defense situations LLMs face can provide feasible research directions, making more areas reap LLMs' benefits.



## **11. Dataset and Lessons Learned from the 2024 SaTML LLM Capture-the-Flag Competition**

cs.CR

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2406.07954v1) [paper-pdf](http://arxiv.org/pdf/2406.07954v1)

**Authors**: Edoardo Debenedetti, Javier Rando, Daniel Paleka, Silaghi Fineas Florin, Dragos Albastroiu, Niv Cohen, Yuval Lemberg, Reshmi Ghosh, Rui Wen, Ahmed Salem, Giovanni Cherubin, Santiago Zanella-Beguelin, Robin Schmid, Victor Klemm, Takahiro Miki, Chenhao Li, Stefan Kraft, Mario Fritz, Florian Tramèr, Sahar Abdelnabi, Lea Schönherr

**Abstract**: Large language model systems face important security risks from maliciously crafted messages that aim to overwrite the system's original instructions or leak private data. To study this problem, we organized a capture-the-flag competition at IEEE SaTML 2024, where the flag is a secret string in the LLM system prompt. The competition was organized in two phases. In the first phase, teams developed defenses to prevent the model from leaking the secret. During the second phase, teams were challenged to extract the secrets hidden for defenses proposed by the other teams. This report summarizes the main insights from the competition. Notably, we found that all defenses were bypassed at least once, highlighting the difficulty of designing a successful defense and the necessity for additional research to protect LLM systems. To foster future research in this direction, we compiled a dataset with over 137k multi-turn attack chats and open-sourced the platform.



## **12. Visual-RolePlay: Universal Jailbreak Attack on MultiModal Large Language Models via Role-playing Image Character**

cs.CR

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2405.20773v2) [paper-pdf](http://arxiv.org/pdf/2405.20773v2)

**Authors**: Siyuan Ma, Weidi Luo, Yu Wang, Xiaogeng Liu

**Abstract**: With the advent and widespread deployment of Multimodal Large Language Models (MLLMs), ensuring their safety has become increasingly critical. To achieve this objective, it requires us to proactively discover the vulnerability of MLLMs by exploring the attack methods. Thus, structure-based jailbreak attacks, where harmful semantic content is embedded within images, have been proposed to mislead the models. However, previous structure-based jailbreak methods mainly focus on transforming the format of malicious queries, such as converting harmful content into images through typography, which lacks sufficient jailbreak effectiveness and generalizability. To address these limitations, we first introduce the concept of "Role-play" into MLLM jailbreak attacks and propose a novel and effective method called Visual Role-play (VRP). Specifically, VRP leverages Large Language Models to generate detailed descriptions of high-risk characters and create corresponding images based on the descriptions. When paired with benign role-play instruction texts, these high-risk character images effectively mislead MLLMs into generating malicious responses by enacting characters with negative attributes. We further extend our VRP method into a universal setup to demonstrate its generalizability. Extensive experiments on popular benchmarks show that VRP outperforms the strongest baseline, Query relevant and FigStep, by an average Attack Success Rate (ASR) margin of 14.3% across all models.



## **13. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

cs.CL

19 Pages, 5 Figures, 8 Tables. Accepted by ACL 2024

**SubmitDate**: 2024-06-12    [abs](http://arxiv.org/abs/2309.14348v3) [paper-pdf](http://arxiv.org/pdf/2309.14348v3)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100% to around 10% or less.



## **14. Merging Improves Self-Critique Against Jailbreak Attacks**

cs.CL

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07188v1) [paper-pdf](http://arxiv.org/pdf/2406.07188v1)

**Authors**: Victor Gallego

**Abstract**: The robustness of large language models (LLMs) against adversarial manipulations, such as jailbreak attacks, remains a significant challenge. In this work, we propose an approach that enhances the self-critique capability of the LLM and further fine-tunes it over sanitized synthetic data. This is done with the addition of an external critic model that can be merged with the original, thus bolstering self-critique capabilities and improving the robustness of the LLMs response to adversarial prompts. Our results demonstrate that the combination of merging and self-critique can reduce the attack success rate of adversaries significantly, thus offering a promising defense mechanism against jailbreak attacks. Code, data and models released at https://github.com/vicgalle/merging-self-critique-jailbreaks .



## **15. Benchmarking Trustworthiness of Multimodal Large Language Models: A Comprehensive Study**

cs.CL

100 pages, 84 figures, 33 tables

**SubmitDate**: 2024-06-11    [abs](http://arxiv.org/abs/2406.07057v1) [paper-pdf](http://arxiv.org/pdf/2406.07057v1)

**Authors**: Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu

**Abstract**: Despite the superior capabilities of Multimodal Large Language Models (MLLMs) across diverse tasks, they still face significant trustworthiness challenges. Yet, current literature on the assessment of trustworthy MLLMs remains limited, lacking a holistic evaluation to offer thorough insights into future improvements. In this work, we establish MultiTrust, the first comprehensive and unified benchmark on the trustworthiness of MLLMs across five primary aspects: truthfulness, safety, robustness, fairness, and privacy. Our benchmark employs a rigorous evaluation strategy that addresses both multimodal risks and cross-modal impacts, encompassing 32 diverse tasks with self-curated datasets. Extensive experiments with 21 modern MLLMs reveal some previously unexplored trustworthiness issues and risks, highlighting the complexities introduced by the multimodality and underscoring the necessity for advanced methodologies to enhance their reliability. For instance, typical proprietary models still struggle with the perception of visually confusing images and are vulnerable to multimodal jailbreaking and adversarial attacks; MLLMs are more inclined to disclose privacy in text and reveal ideological and cultural biases even when paired with irrelevant images in inference, indicating that the multimodality amplifies the internal risks from base LLMs. Additionally, we release a scalable toolbox for standardized trustworthiness research, aiming to facilitate future advancements in this important field. Code and resources are publicly available at: https://multi-trust.github.io/.



## **16. An LLM-Assisted Easy-to-Trigger Backdoor Attack on Code Completion Models: Injecting Disguised Vulnerabilities against Strong Detection**

cs.CR

To appear in USENIX Security '24

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06822v1) [paper-pdf](http://arxiv.org/pdf/2406.06822v1)

**Authors**: Shenao Yan, Shen Wang, Yue Duan, Hanbin Hong, Kiho Lee, Doowon Kim, Yuan Hong

**Abstract**: Large Language Models (LLMs) have transformed code completion tasks, providing context-based suggestions to boost developer productivity in software engineering. As users often fine-tune these models for specific applications, poisoning and backdoor attacks can covertly alter the model outputs. To address this critical security challenge, we introduce CodeBreaker, a pioneering LLM-assisted backdoor attack framework on code completion models. Unlike recent attacks that embed malicious payloads in detectable or irrelevant sections of the code (e.g., comments), CodeBreaker leverages LLMs (e.g., GPT-4) for sophisticated payload transformation (without affecting functionalities), ensuring that both the poisoned data for fine-tuning and generated code can evade strong vulnerability detection. CodeBreaker stands out with its comprehensive coverage of vulnerabilities, making it the first to provide such an extensive set for evaluation. Our extensive experimental evaluations and user studies underline the strong attack performance of CodeBreaker across various settings, validating its superiority over existing approaches. By integrating malicious payloads directly into the source code with minimal transformation, CodeBreaker challenges current security measures, underscoring the critical need for more robust defenses for code completion.



## **17. Better Safe than Sorry: Pre-training CLIP against Targeted Data Poisoning and Backdoor Attacks**

cs.LG

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2310.05862v2) [paper-pdf](http://arxiv.org/pdf/2310.05862v2)

**Authors**: Wenhan Yang, Jingdong Gao, Baharan Mirzasoleiman

**Abstract**: Contrastive Language-Image Pre-training (CLIP) on large image-caption datasets has achieved remarkable success in zero-shot classification and enabled transferability to new domains. However, CLIP is extremely more vulnerable to targeted data poisoning and backdoor attacks, compared to supervised learning. Perhaps surprisingly, poisoning 0.0001% of CLIP pre-training data is enough to make targeted data poisoning attacks successful. This is four orders of magnitude smaller than what is required to poison supervised models. Despite this vulnerability, existing methods are very limited in defending CLIP models during pre-training. In this work, we propose a strong defense, SAFECLIP, to safely pre-train CLIP against targeted data poisoning and backdoor attacks. SAFECLIP warms up the model by applying unimodal contrastive learning (CL) on image and text modalities separately. Then, it divides the data into safe and risky sets, by applying a Gaussian Mixture Model to the cosine similarity of image-caption pair representations. SAFECLIP pre-trains the model by applying the CLIP loss to the safe set and applying unimodal CL to image and text modalities of the risky set separately. By gradually increasing the size of the safe set during pre-training, SAFECLIP effectively breaks targeted data poisoning and backdoor attacks without harming the CLIP performance. Our extensive experiments on CC3M, Visual Genome, and MSCOCO demonstrate that SAFECLIP significantly reduces the success rate of targeted data poisoning attacks from 93.75% to 0% and that of various backdoor attacks from up to 100% to 0%, without harming CLIP's performance.



## **18. LLM Dataset Inference: Did you train on my dataset?**

cs.LG

Code is available at  \href{https://github.com/pratyushmaini/llm_dataset_inference/

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06443v1) [paper-pdf](http://arxiv.org/pdf/2406.06443v1)

**Authors**: Pratyush Maini, Hengrui Jia, Nicolas Papernot, Adam Dziedzic

**Abstract**: The proliferation of large language models (LLMs) in the real world has come with a rise in copyright cases against companies for training their models on unlicensed data from the internet. Recent works have presented methods to identify if individual text sequences were members of the model's training data, known as membership inference attacks (MIAs). We demonstrate that the apparent success of these MIAs is confounded by selecting non-members (text sequences not used for training) belonging to a different distribution from the members (e.g., temporally shifted recent Wikipedia articles compared with ones used to train the model). This distribution shift makes membership inference appear successful. However, most MIA methods perform no better than random guessing when discriminating between members and non-members from the same distribution (e.g., in this case, the same period of time). Even when MIAs work, we find that different MIAs succeed at inferring membership of samples from different distributions. Instead, we propose a new dataset inference method to accurately identify the datasets used to train large language models. This paradigm sits realistically in the modern-day copyright landscape, where authors claim that an LLM is trained over multiple documents (such as a book) written by them, rather than one particular paragraph. While dataset inference shares many of the challenges of membership inference, we solve it by selectively combining the MIAs that provide positive signal for a given distribution, and aggregating them to perform a statistical test on a given dataset. Our approach successfully distinguishes the train and test sets of different subsets of the Pile with statistically significant p-values < 0.1, without any false positives.



## **19. Are you still on track!? Catching LLM Task Drift with Activations**

cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.00799v2) [paper-pdf](http://arxiv.org/pdf/2406.00799v2)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: Large Language Models (LLMs) are routinely used in retrieval-augmented applications to orchestrate tasks and process inputs from users and other sources. These inputs, even in a single LLM interaction, can come from a variety of sources, of varying trustworthiness and provenance. This opens the door to prompt injection attacks, where the LLM receives and acts upon instructions from supposedly data-only sources, thus deviating from the user's original instructions. We define this as task drift, and we propose to catch it by scanning and analyzing the LLM's activations. We compare the LLM's activations before and after processing the external input in order to detect whether this input caused instruction drift. We develop two probing methods and find that simply using a linear classifier can detect drift with near perfect ROC AUC on an out-of-distribution test set. We show that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Our setup does not require any modification of the LLM (e.g., fine-tuning) or any text generation, thus maximizing deployability and cost efficiency and avoiding reliance on unreliable model output. To foster future research on activation-based task inspection, decoding, and interpretability, we will release our large-scale TaskTracker toolkit, comprising a dataset of over 500K instances, representations from 4 SoTA language models, and inspection tools.



## **20. Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction**

cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2402.18104v2) [paper-pdf](http://arxiv.org/pdf/2402.18104v2)

**Authors**: Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, Kai Chen

**Abstract**: In recent years, large language models (LLMs) have demonstrated notable success across various tasks, but the trustworthiness of LLMs is still an open problem. One specific threat is the potential to generate toxic or harmful responses. Attackers can craft adversarial prompts that induce harmful responses from LLMs. In this work, we pioneer a theoretical foundation in LLMs security by identifying bias vulnerabilities within the safety fine-tuning and design a black-box jailbreak method named DRA (Disguise and Reconstruction Attack), which conceals harmful instructions through disguise and prompts the model to reconstruct the original harmful instruction within its completion. We evaluate DRA across various open-source and closed-source models, showcasing state-of-the-art jailbreak success rates and attack efficiency. Notably, DRA boasts a 91.1% attack success rate on OpenAI GPT-4 chatbot.



## **21. CARES: A Comprehensive Benchmark of Trustworthiness in Medical Vision Language Models**

cs.LG

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06007v1) [paper-pdf](http://arxiv.org/pdf/2406.06007v1)

**Authors**: Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou, Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou, Kangyu Zhu, Wenhao Zheng, Zhaoyang Wang, Xiao Wang, Xuchao Zhang, Chetan Bansal, Marc Niethammer, Junzhou Huang, Hongtu Zhu, Yun Li, Jimeng Sun, Zongyuan Ge, Gang Li, James Zou, Huaxiu Yao

**Abstract**: Artificial intelligence has significantly impacted medical applications, particularly with the advent of Medical Large Vision Language Models (Med-LVLMs), sparking optimism for the future of automated and personalized healthcare. However, the trustworthiness of Med-LVLMs remains unverified, posing significant risks for future model deployment. In this paper, we introduce CARES and aim to comprehensively evaluate the Trustworthiness of Med-LVLMs across the medical domain. We assess the trustworthiness of Med-LVLMs across five dimensions, including trustfulness, fairness, safety, privacy, and robustness. CARES comprises about 41K question-answer pairs in both closed and open-ended formats, covering 16 medical image modalities and 27 anatomical regions. Our analysis reveals that the models consistently exhibit concerns regarding trustworthiness, often displaying factual inaccuracies and failing to maintain fairness across different demographic groups. Furthermore, they are vulnerable to attacks and demonstrate a lack of privacy awareness. We publicly release our benchmark and code in https://github.com/richard-peng-xia/CARES.



## **22. Chain-of-Scrutiny: Detecting Backdoor Attacks for Large Language Models**

cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.05948v1) [paper-pdf](http://arxiv.org/pdf/2406.05948v1)

**Authors**: Xi Li, Yusen Zhang, Renze Lou, Chen Wu, Jiaqi Wang

**Abstract**: Backdoor attacks present significant threats to Large Language Models (LLMs), particularly with the rise of third-party services that offer API integration and prompt engineering. Untrustworthy third parties can plant backdoors into LLMs and pose risks to users by embedding malicious instructions into user queries. The backdoor-compromised LLM will generate malicious output when and input is embedded with a specific trigger predetermined by an attacker. Traditional defense strategies, which primarily involve model parameter fine-tuning and gradient calculation, are inadequate for LLMs due to their extensive computational and clean data requirements. In this paper, we propose a novel solution, Chain-of-Scrutiny (CoS), to address these challenges. Backdoor attacks fundamentally create a shortcut from the trigger to the target output, thus lack reasoning support. Accordingly, CoS guides the LLMs to generate detailed reasoning steps for the input, then scrutinizes the reasoning process to ensure consistency with the final answer. Any inconsistency may indicate an attack. CoS only requires black-box access to LLM, offering a practical defense, particularly for API-accessible LLMs. It is user-friendly, enabling users to conduct the defense themselves. Driven by natural language, the entire defense process is transparent to users. We validate the effectiveness of CoS through extensive experiments across various tasks and LLMs. Additionally, experiments results shows CoS proves more beneficial for more powerful LLMs.



## **23. Safety Alignment Should Be Made More Than Just a Few Tokens Deep**

cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.05946v1) [paper-pdf](http://arxiv.org/pdf/2406.05946v1)

**Authors**: Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, Peter Henderson

**Abstract**: The safety alignment of current Large Language Models (LLMs) is vulnerable. Relatively simple attacks, or even benign fine-tuning, can jailbreak aligned models. We argue that many of these vulnerabilities are related to a shared underlying issue: safety alignment can take shortcuts, wherein the alignment adapts a model's generative distribution primarily over only its very first few output tokens. We refer to this issue as shallow safety alignment. In this paper, we present case studies to explain why shallow safety alignment can exist and provide evidence that current aligned LLMs are subject to this issue. We also show how these findings help explain multiple recently discovered vulnerabilities in LLMs, including the susceptibility to adversarial suffix attacks, prefilling attacks, decoding parameter attacks, and fine-tuning attacks. Importantly, we discuss how this consolidated notion of shallow safety alignment sheds light on promising research directions for mitigating these vulnerabilities. For instance, we show that deepening the safety alignment beyond just the first few tokens can often meaningfully improve robustness against some common exploits. Finally, we design a regularized finetuning objective that makes the safety alignment more persistent against fine-tuning attacks by constraining updates on initial tokens. Overall, we advocate that future safety alignment should be made more than just a few tokens deep.



## **24. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

cs.LG

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2402.06255v2) [paper-pdf](http://arxiv.org/pdf/2402.06255v2)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreak attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly with a particular focus on harmful content filtering or heuristical defensive prompt designs. However, how to achieve intrinsic robustness through the prompts remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both black-box and white-box attacks, reducing the success rate of advanced attacks to nearly 0 while maintaining the model's utility on the benign task. The proposed defense strategy incurs only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/rain152/PAT.



## **25. Adaptive Text Watermark for Large Language Models**

cs.CL

ICML2024

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2401.13927v2) [paper-pdf](http://arxiv.org/pdf/2401.13927v2)

**Authors**: Yepeng Liu, Yuheng Bu

**Abstract**: The advancement of Large Language Models (LLMs) has led to increasing concerns about the misuse of AI-generated text, and watermarking for LLM-generated text has emerged as a potential solution. However, it is challenging to generate high-quality watermarked text while maintaining strong security, robustness, and the ability to detect watermarks without prior knowledge of the prompt or model. This paper proposes an adaptive watermarking strategy to address this problem. To improve the text quality and maintain robustness, we adaptively add watermarking to token distributions with high entropy measured using an auxiliary model and keep the low entropy token distributions untouched. For the sake of security and to further minimize the watermark's impact on text quality, instead of using a fixed green/red list generated from a random secret key, which can be vulnerable to decryption and forgery, we adaptively scale up the output logits in proportion based on the semantic embedding of previously generated text using a well designed semantic mapping model. Our experiments involving various LLMs demonstrate that our approach achieves comparable robustness performance to existing watermark methods. Additionally, the text generated by our method has perplexity comparable to that of \emph{un-watermarked} LLMs while maintaining security even under various attacks.



## **26. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

cs.CR

This paper completes its earlier vision paper, available at  arXiv:2402.15727

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2406.05498v1) [paper-pdf](http://arxiv.org/pdf/2406.05498v1)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into four major categories: optimization-based attacks such as Greedy Coordinate Gradient (GCG), jailbreak template-based attacks such as "Do-Anything-Now", advanced indirect attacks like DrAttack, and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delay to user prompts, as well as be compatible with both open-source and closed-source LLMs.   Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM defense instance to concurrently protect the target LLM instance in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs (both target and defense LLMs) have the capability to identify harmful prompts or intentions in user queries, which we empirically validate using the commonly used GPT-3.5/4 models across all major jailbreak attacks. Our measurements show that SelfDefend enables GPT-3.5 to suppress the attack success rate (ASR) by 8.97-95.74% (average: 60%) and GPT-4 by even 36.36-100% (average: 83%), while incurring negligible effects on normal queries. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. These models outperform four SOTA defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. We also empirically show that the tuned models are robust to targeted GCG and prompt injection attacks.



## **27. MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models**

cs.CV

The datasets were incomplete as they did not include all the  necessary copyrights

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2311.17600v3) [paper-pdf](http://arxiv.org/pdf/2311.17600v3)

**Authors**: Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Multimodal Large Language Models (MLLMs) remains understudied. In this paper, we observe that Multimodal Large Language Models (MLLMs) can be easily compromised by query-relevant images, as if the text query itself were malicious. To address this, we introduce MM-SafetyBench, a comprehensive framework designed for conducting safety-critical evaluations of MLLMs against such image-based manipulations. We have compiled a dataset comprising 13 scenarios, resulting in a total of 5,040 text-image pairs. Our analysis across 12 state-of-the-art models reveals that MLLMs are susceptible to breaches instigated by our approach, even when the equipped LLMs have been safety-aligned. In response, we propose a straightforward yet effective prompting strategy to enhance the resilience of MLLMs against these types of attacks. Our work underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source MLLMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **28. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

cs.CV

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2406.05491v1) [paper-pdf](http://arxiv.org/pdf/2406.05491v1)

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models trained on large-scale image-text pairs have demonstrated unprecedented capability in many practical applications. However, previous studies have revealed that VLP models are vulnerable to adversarial samples crafted by a malicious adversary. While existing attacks have achieved great success in improving attack effect and transferability, they all focus on instance-specific attacks that generate perturbations for each input sample. In this paper, we show that VLP models can be vulnerable to a new class of universal adversarial perturbation (UAP) for all input samples. Although initially transplanting existing UAP algorithms to perform attacks showed effectiveness in attacking discriminative models, the results were unsatisfactory when applied to VLP models. To this end, we revisit the multimodal alignments in VLP model training and propose the Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC). Specifically, we first design a generator that incorporates cross-modal information as conditioning input to guide the training. To further exploit cross-modal interactions, we propose to formulate the training objective as a multimodal contrastive learning paradigm based on our constructed positive and negative image-text pairs. By training the conditional generator with the designed loss, we successfully force the adversarial samples to move away from its original area in the VLP model's feature space, and thus essentially enhance the attacks. Extensive experiments show that our method achieves remarkable attack performance across various VLP models and Vision-and-Language (V+L) tasks. Moreover, C-PGC exhibits outstanding black-box transferability and achieves impressive results in fooling prevalent large VLP models including LLaVA and Qwen-VL.



## **29. PRSA: PRompt Stealing Attacks against Large Language Models**

cs.CR

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2402.19200v2) [paper-pdf](http://arxiv.org/pdf/2402.19200v2)

**Authors**: Yong Yang, Changjiang Li, Yi Jiang, Xi Chen, Haoyu Wang, Xuhong Zhang, Zonghui Wang, Shouling Ji

**Abstract**: In recent years, "prompt as a service" has greatly enhanced the utility of large language models (LLMs) by enabling them to perform various downstream tasks efficiently without fine-tuning. This has also increased the commercial value of prompts. However, the potential risk of leakage in these commercialized prompts remains largely underexplored. In this paper, we introduce a novel attack framework, PRSA, designed for prompt stealing attacks against LLMs. The main idea of PRSA is to infer the intent behind a prompt by analyzing its input-output content, enabling the generation of a surrogate prompt that replicates the original's functionality. Specifically, PRSA mainly consists of two key phases: prompt mutation and prompt pruning. In the mutation phase, we propose a prompt attention algorithm based on output difference. The algorithm facilitates the generation of effective surrogate prompts by learning key factors that influence the accurate inference of prompt intent. During the pruning phase, we employ a two-step related word identification strategy to detect and mask words that are highly related to the input, thus improving the generalizability of the surrogate prompts. We verify the actual threat of PRSA through evaluation in both real-world settings, non-interactive and interactive prompt services. The results strongly confirm the PRSA's effectiveness and generalizability. We have reported these findings to prompt service providers and actively collaborate with them to implement defensive measures.



## **30. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

cs.CR

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.03230v2) [paper-pdf](http://arxiv.org/pdf/2406.03230v2)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.



## **31. ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs**

cs.CL

To appear in ACL 2024

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.11753v4) [paper-pdf](http://arxiv.org/pdf/2402.11753v4)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Zhen Xiang, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran

**Abstract**: Safety is critical to the usage of large language models (LLMs). Multiple techniques such as data filtering and supervised fine-tuning have been developed to strengthen LLM safety. However, currently known techniques presume that corpora used for safety alignment of LLMs are solely interpreted by semantics. This assumption, however, does not hold in real-world applications, which leads to severe vulnerabilities in LLMs. For example, users of forums often use ASCII art, a form of text-based art, to convey image information. In this paper, we propose a novel ASCII art-based jailbreak attack and introduce a comprehensive benchmark Vision-in-Text Challenge (ViTC) to evaluate the capabilities of LLMs in recognizing prompts that cannot be solely interpreted by semantics. We show that five SOTA LLMs (GPT-3.5, GPT-4, Gemini, Claude, and Llama2) struggle to recognize prompts provided in the form of ASCII art. Based on this observation, we develop the jailbreak attack ArtPrompt, which leverages the poor performance of LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs. ArtPrompt only requires black-box access to the victim LLMs, making it a practical attack. We evaluate ArtPrompt on five SOTA LLMs, and show that ArtPrompt can effectively and efficiently induce undesired behaviors from all five LLMs. Our code is available at https://github.com/uw-nsl/ArtPrompt.



## **32. SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding**

cs.CR

To appear in ACL 2024

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.08983v3) [paper-pdf](http://arxiv.org/pdf/2402.08983v3)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, Radha Poovendran

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their token probabilities, while simultaneously attenuating the probabilities of token sequences that are aligned with the objectives of jailbreak attacks. We perform extensive experiments on five LLMs using six state-of-the-art jailbreak attacks and four benchmark datasets. Our results show that SafeDecoding significantly reduces the attack success rate and harmfulness of jailbreak attacks without compromising the helpfulness of responses to benign user queries. SafeDecoding outperforms six defense methods.



## **33. DepsRAG: Towards Managing Software Dependencies using Large Language Models**

cs.SE

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2405.20455v3) [paper-pdf](http://arxiv.org/pdf/2405.20455v3)

**Authors**: Mohannad Alhanahnah, Yazan Boshmaf, Benoit Baudry

**Abstract**: Managing software dependencies is a crucial maintenance task in software development and is becoming a rapidly growing research field, especially in light of the significant increase in software supply chain attacks. Specialized expertise and substantial developer effort are required to fully comprehend dependencies and reveal hidden properties about the dependencies (e.g., number of dependencies, dependency chains, depth of dependencies).   Recent advancements in Large Language Models (LLMs) allow the retrieval of information from various data sources for response generation, thus providing a new opportunity to uniquely manage software dependencies. To highlight the potential of this technology, we present~\tool, a proof-of-concept Retrieval Augmented Generation (RAG) approach that constructs direct and transitive dependencies of software packages as a Knowledge Graph (KG) in four popular software ecosystems. DepsRAG can answer user questions about software dependencies by automatically generating necessary queries to retrieve information from the KG, and then augmenting the input of LLMs with the retrieved information. DepsRAG can also perform Web search to answer questions that the LLM cannot directly answer via the KG. We identify tangible benefits that DepsRAG can offer and discuss its limitations.



## **34. Adversarial Tuning: Defending Against Jailbreak Attacks for LLMs**

cs.CL

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.06622v1) [paper-pdf](http://arxiv.org/pdf/2406.06622v1)

**Authors**: Fan Liu, Zhao Xu, Hao Liu

**Abstract**: Although safely enhanced Large Language Models (LLMs) have achieved remarkable success in tackling various complex tasks in a zero-shot manner, they remain susceptible to jailbreak attacks, particularly the unknown jailbreak attack. To enhance LLMs' generalized defense capabilities, we propose a two-stage adversarial tuning framework, which generates adversarial prompts to explore worst-case scenarios by optimizing datasets containing pairs of adversarial prompts and their safe responses. In the first stage, we introduce the hierarchical meta-universal adversarial prompt learning to efficiently and effectively generate token-level adversarial prompts. In the second stage, we propose the automatic adversarial prompt learning to iteratively refine semantic-level adversarial prompts, further enhancing LLM's defense capabilities. We conducted comprehensive experiments on three widely used jailbreak datasets, comparing our framework with six defense baselines under five representative attack scenarios. The results underscore the superiority of our proposed methods. Furthermore, our adversarial tuning framework exhibits empirical generalizability across various attack strategies and target LLMs, highlighting its potential as a transferable defense mechanism.



## **35. SALAD-Bench: A Hierarchical and Comprehensive Safety Benchmark for Large Language Models**

cs.CL

Accepted at ACL 2024 Findings

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.05044v4) [paper-pdf](http://arxiv.org/pdf/2402.05044v4)

**Authors**: Lijun Li, Bowen Dong, Ruohui Wang, Xuhao Hu, Wangmeng Zuo, Dahua Lin, Yu Qiao, Jing Shao

**Abstract**: In the rapidly evolving landscape of Large Language Models (LLMs), ensuring robust safety measures is paramount. To meet this crucial need, we propose \emph{SALAD-Bench}, a safety benchmark specifically designed for evaluating LLMs, attack, and defense methods. Distinguished by its breadth, SALAD-Bench transcends conventional benchmarks through its large scale, rich diversity, intricate taxonomy spanning three levels, and versatile functionalities.SALAD-Bench is crafted with a meticulous array of questions, from standard queries to complex ones enriched with attack, defense modifications and multiple-choice. To effectively manage the inherent complexity, we introduce an innovative evaluators: the LLM-based MD-Judge for QA pairs with a particular focus on attack-enhanced queries, ensuring a seamless, and reliable evaluation. Above components extend SALAD-Bench from standard LLM safety evaluation to both LLM attack and defense methods evaluation, ensuring the joint-purpose utility. Our extensive experiments shed light on the resilience of LLMs against emerging threats and the efficacy of contemporary defense tactics. Data and evaluator are released under https://github.com/OpenSafetyLab/SALAD-BENCH.



## **36. Sales Whisperer: A Human-Inconspicuous Attack on LLM Brand Recommendations**

cs.CR

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.04755v1) [paper-pdf](http://arxiv.org/pdf/2406.04755v1)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Large language model (LLM) users might rely on others (e.g., prompting services), to write prompts. However, the risks of trusting prompts written by others remain unstudied. In this paper, we assess the risk of using such prompts on brand recommendation tasks when shopping. First, we found that paraphrasing prompts can result in LLMs mentioning given brands with drastically different probabilities, including a pair of prompts where the probability changes by 100%. Next, we developed an approach that can be used to perturb an original base prompt to increase the likelihood that an LLM mentions a given brand. We designed a human-inconspicuous algorithm that perturbs prompts, which empirically forces LLMs to mention strings related to a brand more often, by absolute improvements up to 78.3%. Our results suggest that our perturbed prompts, 1) are inconspicuous to humans, 2) force LLMs to recommend a target brand more often, and 3) increase the perceived chances of picking targeted brands.



## **37. COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability**

cs.LG

Accepted to ICML 2024

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.08679v2) [paper-pdf](http://arxiv.org/pdf/2402.08679v2)

**Authors**: Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, Bin Hu

**Abstract**: Jailbreaks on large language models (LLMs) have recently received increasing attention. For a comprehensive assessment of LLM safety, it is essential to consider jailbreaks with diverse attributes, such as contextual coherence and sentiment/stylistic variations, and hence it is beneficial to study controllable jailbreaking, i.e. how to enforce control on LLM attacks. In this paper, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the Energy-based Constrained Decoding with Langevin Dynamics (COLD), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios which not only cover the standard setting of generating fluent (suffix) attack with continuation constraint, but also allow us to address new controllable attack settings such as revising a user query adversarially with paraphrasing constraint, and inserting stealthy attacks in context with position constraint. Our extensive experiments on various LLMs (Llama-2, Mistral, Vicuna, Guanaco, GPT-3.5, and GPT-4) show COLD-Attack's broad applicability, strong controllability, high success rate, and attack transferability. Our code is available at https://github.com/Yu-Fangxu/COLD-Attack.



## **38. Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack**

cs.CL

Accepted to ACL2024 main

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2312.06924v2) [paper-pdf](http://arxiv.org/pdf/2312.06924v2)

**Authors**: Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong

**Abstract**: Recent developments in balancing the usefulness and safety of Large Language Models (LLMs) have raised a critical question: Are mainstream NLP tasks adequately aligned with safety consideration? Our study, focusing on safety-sensitive documents obtained through adversarial attacks, reveals significant disparities in the safety alignment of various NLP tasks. For instance, LLMs can effectively summarize malicious long documents but often refuse to translate them. This discrepancy highlights a previously unidentified vulnerability: attacks exploiting tasks with weaker safety alignment, like summarization, can potentially compromise the integrity of tasks traditionally deemed more robust, such as translation and question-answering (QA). Moreover, the concurrent use of multiple NLP tasks with lesser safety alignment increases the risk of LLMs inadvertently processing harmful content. We demonstrate these vulnerabilities in various safety-aligned LLMs, particularly Llama2 models, Gemini and GPT-4, indicating an urgent need for strengthening safety alignments across a broad spectrum of NLP tasks.



## **39. Evaluating the Efficacy of Large Language Models in Identifying Phishing Attempts**

cs.CL

7 pages, 3 figures

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2404.15485v3) [paper-pdf](http://arxiv.org/pdf/2404.15485v3)

**Authors**: Het Patel, Umair Rehman, Farkhund Iqbal

**Abstract**: Phishing, a prevalent cybercrime tactic for decades, remains a significant threat in today's digital world. By leveraging clever social engineering elements and modern technology, cybercrime targets many individuals, businesses, and organizations to exploit trust and security. These cyber-attackers are often disguised in many trustworthy forms to appear as legitimate sources. By cleverly using psychological elements like urgency, fear, social proof, and other manipulative strategies, phishers can lure individuals into revealing sensitive and personalized information. Building on this pervasive issue within modern technology, this paper aims to analyze the effectiveness of 15 Large Language Models (LLMs) in detecting phishing attempts, specifically focusing on a randomized set of "419 Scam" emails. The objective is to determine which LLMs can accurately detect phishing emails by analyzing a text file containing email metadata based on predefined criteria. The experiment concluded that the following models, ChatGPT 3.5, GPT-3.5-Turbo-Instruct, and ChatGPT, were the most effective in detecting phishing emails.



## **40. Defending LLMs against Jailbreaking Attacks via Backtranslation**

cs.CL

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2402.16459v3) [paper-pdf](http://arxiv.org/pdf/2402.16459v3)

**Authors**: Yihan Wang, Zhouxing Shi, Andrew Bai, Cho-Jui Hsieh

**Abstract**: Although many large language models (LLMs) have been trained to refuse harmful requests, they are still vulnerable to jailbreaking attacks which rewrite the original prompt to conceal its harmful intent. In this paper, we propose a new method for defending LLMs against jailbreaking attacks by ``backtranslation''. Specifically, given an initial response generated by the target LLM from an input prompt, our backtranslation prompts a language model to infer an input prompt that can lead to the response. The inferred prompt is called the backtranslated prompt which tends to reveal the actual intent of the original prompt, since it is generated based on the LLM's response and not directly manipulated by the attacker. We then run the target LLM again on the backtranslated prompt, and we refuse the original prompt if the model refuses the backtranslated prompt. We explain that the proposed defense provides several benefits on its effectiveness and efficiency. We empirically demonstrate that our defense significantly outperforms the baselines, in the cases that are hard for the baselines, and our defense also has little impact on the generation quality for benign input prompts. Our implementation is based on our library for LLM jailbreaking defense algorithms at \url{https://github.com/YihanWang617/llm-jailbreaking-defense}, and the code for reproducing our experiments is available at \url{https://github.com/YihanWang617/LLM-Jailbreaking-Defense-Backtranslation}.



## **41. BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models**

cs.CR

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.00083v2) [paper-pdf](http://arxiv.org/pdf/2406.00083v2)

**Authors**: Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, Qian Lou

**Abstract**: Large Language Models (LLMs) are constrained by outdated information and a tendency to generate incorrect data, commonly referred to as "hallucinations." Retrieval-Augmented Generation (RAG) addresses these limitations by combining the strengths of retrieval-based methods and generative models. This approach involves retrieving relevant information from a large, up-to-date dataset and using it to enhance the generation process, leading to more accurate and contextually appropriate responses. Despite its benefits, RAG introduces a new attack surface for LLMs, particularly because RAG databases are often sourced from public data, such as the web. In this paper, we propose \TrojRAG{} to identify the vulnerabilities and attacks on retrieval parts (RAG database) and their indirect attacks on generative parts (LLMs). Specifically, we identify that poisoning several customized content passages could achieve a retrieval backdoor, where the retrieval works well for clean queries but always returns customized poisoned adversarial queries. Triggers and poisoned passages can be highly customized to implement various attacks. For example, a trigger could be a semantic group like "The Republican Party, Donald Trump, etc." Adversarial passages can be tailored to different contents, not only linked to the triggers but also used to indirectly attack generative LLMs without modifying them. These attacks can include denial-of-service attacks on RAG and semantic steering attacks on LLM generations conditioned by the triggers. Our experiments demonstrate that by just poisoning 10 adversarial passages can induce 98.2\% success rate to retrieve the adversarial passages. Then, these passages can increase the reject ratio of RAG-based GPT-4 from 0.01\% to 74.6\% or increase the rate of negative responses from 0.22\% to 72\% for targeted queries.



## **42. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

cs.CV

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04031v1) [paper-pdf](http://arxiv.org/pdf/2406.04031v1)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.



## **43. Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!**

cs.CL

ACL 2024

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2402.12343v4) [paper-pdf](http://arxiv.org/pdf/2402.12343v4)

**Authors**: Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao

**Abstract**: Large language models (LLMs) undergo safety alignment to ensure safe conversations with humans. However, this paper introduces a training-free attack method capable of reversing safety alignment, converting the outcomes of stronger alignment into greater potential for harm by accessing only LLM output token distributions. Specifically, our method achieves this reversal by contrasting the output token distribution of a safety-aligned language model (e.g., Llama-2-chat) against its pre-trained version (e.g., Llama-2), so that the token predictions are shifted towards the opposite direction of safety alignment. We name this method emulated disalignment (ED) because sampling from this contrastive distribution provably emulates the result of fine-tuning to minimize a safety reward. Our experiments with ED across three evaluation datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show that ED doubles the harmfulness of pre-trained models and outperforms strong baselines, achieving the highest harmful rates in 43 out of 48 evaluation subsets by a large margin. Eventually, given ED's reliance on language model output token distributions, which particularly compromises open-source models, our findings highlight the need to reassess the open accessibility of language models, even if they have been safety-aligned. Code is available at https://github.com/ZHZisZZ/emulated-disalignment.



## **44. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

cs.CL

Competition Report

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2404.14461v2) [paper-pdf](http://arxiv.org/pdf/2404.14461v2)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.



## **45. AutoJailbreak: Exploring Jailbreak Attacks and Defenses through a Dependency Lens**

cs.CR

32 pages, 2 figures

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.03805v1) [paper-pdf](http://arxiv.org/pdf/2406.03805v1)

**Authors**: Lin Lu, Hai Yan, Zenghui Yuan, Jiawen Shi, Wenqi Wei, Pin-Yu Chen, Pan Zhou

**Abstract**: Jailbreak attacks in large language models (LLMs) entail inducing the models to generate content that breaches ethical and legal norm through the use of malicious prompts, posing a substantial threat to LLM security. Current strategies for jailbreak attack and defense often focus on optimizing locally within specific algorithmic frameworks, resulting in ineffective optimization and limited scalability. In this paper, we present a systematic analysis of the dependency relationships in jailbreak attack and defense techniques, generalizing them to all possible attack surfaces. We employ directed acyclic graphs (DAGs) to position and analyze existing jailbreak attacks, defenses, and evaluation methodologies, and propose three comprehensive, automated, and logical frameworks. \texttt{AutoAttack} investigates dependencies in two lines of jailbreak optimization strategies: genetic algorithm (GA)-based attacks and adversarial-generation-based attacks, respectively. We then introduce an ensemble jailbreak attack to exploit these dependencies. \texttt{AutoDefense} offers a mixture-of-defenders approach by leveraging the dependency relationships in pre-generative and post-generative defense strategies. \texttt{AutoEvaluation} introduces a novel evaluation method that distinguishes hallucinations, which are often overlooked, from jailbreak attack and defense responses. Through extensive experiments, we demonstrate that the proposed ensemble jailbreak attack and defense framework significantly outperforms existing research.



## **46. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

cs.LG

Code available at https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2401.17263v3) [paper-pdf](http://arxiv.org/pdf/2401.17263v3)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo



## **47. Stealthy Attack on Large Language Model based Recommendation**

cs.CL

ACL 2024 Main

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.14836v2) [paper-pdf](http://arxiv.org/pdf/2402.14836v2)

**Authors**: Jinghao Zhang, Yuting Liu, Qiang Liu, Shu Wu, Guibing Guo, Liang Wang

**Abstract**: Recently, the powerful large language models (LLMs) have been instrumental in propelling the progress of recommender systems (RS). However, while these systems have flourished, their susceptibility to security threats has been largely overlooked. In this work, we reveal that the introduction of LLMs into recommendation models presents new security vulnerabilities due to their emphasis on the textual content of items. We demonstrate that attackers can significantly boost an item's exposure by merely altering its textual content during the testing phase, without requiring direct interference with the model's training process. Additionally, the attack is notably stealthy, as it does not affect the overall recommendation performance and the modifications to the text are subtle, making it difficult for users and platforms to detect. Our comprehensive experiments across four mainstream LLM-based recommendation models demonstrate the superior efficacy and stealthiness of our approach. Our work unveils a significant security gap in LLM-based recommendation systems and paves the way for future research on protecting these systems.



## **48. Improved Techniques for Optimization-Based Jailbreaking on Large Language Models**

cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.21018v2) [paper-pdf](http://arxiv.org/pdf/2405.21018v2)

**Authors**: Xiaojun Jia, Tianyu Pang, Chao Du, Yihao Huang, Jindong Gu, Yang Liu, Xiaochun Cao, Min Lin

**Abstract**: Large language models (LLMs) are being rapidly developed, and a key component of their widespread deployment is their safety-related alignment. Many red-teaming efforts aim to jailbreak LLMs, where among these efforts, the Greedy Coordinate Gradient (GCG) attack's success has led to a growing interest in the study of optimization-based jailbreaking techniques. Although GCG is a significant milestone, its attacking efficiency remains unsatisfactory. In this paper, we present several improved (empirical) techniques for optimization-based jailbreaks like GCG. We first observe that the single target template of "Sure" largely limits the attacking performance of GCG; given this, we propose to apply diverse target templates containing harmful self-suggestion and/or guidance to mislead LLMs. Besides, from the optimization aspects, we propose an automatic multi-coordinate updating strategy in GCG (i.e., adaptively deciding how many tokens to replace in each step) to accelerate convergence, as well as tricks like easy-to-hard initialisation. Then, we combine these improved technologies to develop an efficient jailbreak method, dubbed I-GCG. In our experiments, we evaluate on a series of benchmarks (such as NeurIPS 2023 Red Teaming Track). The results demonstrate that our improved techniques can help GCG outperform state-of-the-art jailbreaking attacks and achieve nearly 100% attack success rate. The code is released at https://github.com/jiaxiaojunQAQ/I-GCG.



## **49. CR-UTP: Certified Robustness against Universal Text Perturbations on Large Language Models**

cs.CL

Accepted by ACL Findings 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.01873v2) [paper-pdf](http://arxiv.org/pdf/2406.01873v2)

**Authors**: Qian Lou, Xin Liang, Jiaqi Xue, Yancheng Zhang, Rui Xie, Mengxin Zheng

**Abstract**: It is imperative to ensure the stability of every prediction made by a language model; that is, a language's prediction should remain consistent despite minor input variations, like word substitutions. In this paper, we investigate the problem of certifying a language model's robustness against Universal Text Perturbations (UTPs), which have been widely used in universal adversarial attacks and backdoor attacks. Existing certified robustness based on random smoothing has shown considerable promise in certifying the input-specific text perturbations (ISTPs), operating under the assumption that any random alteration of a sample's clean or adversarial words would negate the impact of sample-wise perturbations. However, with UTPs, masking only the adversarial words can eliminate the attack. A naive method is to simply increase the masking ratio and the likelihood of masking attack tokens, but it leads to a significant reduction in both certified accuracy and the certified radius due to input corruption by extensive masking. To solve this challenge, we introduce a novel approach, the superior prompt search method, designed to identify a superior prompt that maintains higher certified accuracy under extensive masking. Additionally, we theoretically motivate why ensembles are a particularly suitable choice as base prompts for random smoothing. The method is denoted by superior prompt ensembling technique. We also empirically confirm this technique, obtaining state-of-the-art results in multiple settings. These methodologies, for the first time, enable high certified accuracy against both UTPs and ISTPs. The source code of CR-UTP is available at \url {https://github.com/UCFML-Research/CR-UTP}.



## **50. Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models**

cs.LG

ICML 2024 Oral

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.12336v2) [paper-pdf](http://arxiv.org/pdf/2402.12336v2)

**Authors**: Christian Schlarmann, Naman Deep Singh, Francesco Croce, Matthias Hein

**Abstract**: Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many large vision-language models (LVLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (LVLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of LVLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP model with our robust one. No retraining or fine-tuning of the down-stream LVLMs is required. The code and robust models are available at https://github.com/chs20/RobustVLM



