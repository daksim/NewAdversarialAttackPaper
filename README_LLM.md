# Latest Large Language Model Attack Papers
**update at 2024-12-11 10:17:21**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. SQL Injection Jailbreak: a structural disaster of large language models**

cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2411.01565v3) [paper-pdf](http://arxiv.org/pdf/2411.01565v3)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality into various domains, generating substantial social and economic benefits. However, this swift advancement has also introduced new security vulnerabilities. Jailbreaking, a form of attack that induces LLMs to produce harmful content through carefully crafted prompts, presents a significant challenge to the safe and trustworthy development of LLMs. Previous jailbreak methods primarily exploited the internal properties or capabilities of LLMs, such as optimization-based jailbreak approaches and methods that leveraged the model's context-learning abilities. In this paper, we introduce a novel jailbreak method, SQL Injection Jailbreak (SIJ), which targets the external properties of LLMs, specifically, the way LLMs construct input prompts. By injecting jailbreak information into user prompts, SIJ successfully induces the model to output harmful content. Our SIJ method achieves near 100\% attack success rates on five well-known open-source LLMs on the AdvBench, while incurring lower time costs compared to previous methods. More importantly, SIJ is the first method to exploit the external properties of LLMs for jailbreak attacks and exposes a new vulnerability in LLMs that urgently requires mitigation. To address this, we propose a simple defense method called Self-Reminder-Key to counter SIJ and demonstrate its effectiveness through experimental results. Our code is available at \href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.



## **2. MobileSafetyBench: Evaluating Safety of Autonomous Agents in Mobile Device Control**

cs.LG

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2410.17520v2) [paper-pdf](http://arxiv.org/pdf/2410.17520v2)

**Authors**: Juyong Lee, Dongyoon Hahm, June Suk Choi, W. Bradley Knox, Kimin Lee

**Abstract**: Autonomous agents powered by large language models (LLMs) show promising potential in assistive tasks across various domains, including mobile device control. As these agents interact directly with personal information and device settings, ensuring their safe and reliable behavior is crucial to prevent undesirable outcomes. However, no benchmark exists for standardized evaluation of the safety of mobile device-control agents. In this work, we introduce MobileSafetyBench, a benchmark designed to evaluate the safety of device-control agents within a realistic mobile environment based on Android emulators. We develop a diverse set of tasks involving interactions with various mobile applications, including messaging and banking applications, challenging agents with managing risks encompassing misuse and negative side effects. These tasks include tests to evaluate the safety of agents in daily scenarios as well as their robustness against indirect prompt injection attacks. Our experiments demonstrate that baseline agents, based on state-of-the-art LLMs, often fail to effectively prevent harm while performing the tasks. To mitigate these safety concerns, we propose a prompting method that encourages agents to prioritize safety considerations. While this method shows promise in promoting safer behaviors, there is still considerable room for improvement to fully earn user trust. This highlights the urgent need for continued research to develop more robust safety mechanisms in mobile environments. We open-source our benchmark at: https://mobilesafetybench.github.io/.



## **3. PrisonBreak: Jailbreaking Large Language Models with Fewer Than Twenty-Five Targeted Bit-flips**

cs.CR

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07192v1) [paper-pdf](http://arxiv.org/pdf/2412.07192v1)

**Authors**: Zachary Coalson, Jeonghyun Woo, Shiyang Chen, Yu Sun, Lishan Yang, Prashant Nair, Bo Fang, Sanghyun Hong

**Abstract**: We introduce a new class of attacks on commercial-scale (human-aligned) language models that induce jailbreaking through targeted bitwise corruptions in model parameters. Our adversary can jailbreak billion-parameter language models with fewer than 25 bit-flips in all cases$-$and as few as 5 in some$-$using up to 40$\times$ less bit-flips than existing attacks on computer vision models at least 100$\times$ smaller. Unlike prompt-based jailbreaks, our attack renders these models in memory 'uncensored' at runtime, allowing them to generate harmful responses without any input modifications. Our attack algorithm efficiently identifies target bits to flip, offering up to 20$\times$ more computational efficiency than previous methods. This makes it practical for language models with billions of parameters. We show an end-to-end exploitation of our attack using software-induced fault injection, Rowhammer (RH). Our work examines 56 DRAM RH profiles from DDR4 and LPDDR4X devices with different RH vulnerabilities. We show that our attack can reliably induce jailbreaking in systems similar to those affected by prior bit-flip attacks. Moreover, our approach remains effective even against highly RH-secure systems (e.g., 46$\times$ more secure than previously tested systems). Our analyses further reveal that: (1) models with less post-training alignment require fewer bit flips to jailbreak; (2) certain model components, such as value projection layers, are substantially more vulnerable than others; and (3) our method is mechanistically different than existing jailbreaks. Our findings highlight a pressing, practical threat to the language model ecosystem and underscore the need for research to protect these models from bit-flip attacks.



## **4. Defensive Dual Masking for Robust Adversarial Defense**

cs.CL

First version

**SubmitDate**: 2024-12-10    [abs](http://arxiv.org/abs/2412.07078v1) [paper-pdf](http://arxiv.org/pdf/2412.07078v1)

**Authors**: Wangli Yang, Jie Yang, Yi Guo, Johan Barthelemy

**Abstract**: The field of textual adversarial defenses has gained considerable attention in recent years due to the increasing vulnerability of natural language processing (NLP) models to adversarial attacks, which exploit subtle perturbations in input text to deceive models. This paper introduces the Defensive Dual Masking (DDM) algorithm, a novel approach designed to enhance model robustness against such attacks. DDM utilizes a unique adversarial training strategy where [MASK] tokens are strategically inserted into training samples to prepare the model to handle adversarial perturbations more effectively. During inference, potentially adversarial tokens are dynamically replaced with [MASK] tokens to neutralize potential threats while preserving the core semantics of the input. The theoretical foundation of our approach is explored, demonstrating how the selective masking mechanism strengthens the model's ability to identify and mitigate adversarial manipulations. Our empirical evaluation across a diverse set of benchmark datasets and attack mechanisms consistently shows that DDM outperforms state-of-the-art defense techniques, improving model accuracy and robustness. Moreover, when applied to Large Language Models (LLMs), DDM also enhances their resilience to adversarial attacks, providing a scalable defense mechanism for large-scale NLP applications.



## **5. Unseen Attack Detection in Software-Defined Networking Using a BERT-Based Large Language Model**

cs.CR

Mohammed N. Swileh is first author. Shengli Zhang is corresponding  author

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06239v1) [paper-pdf](http://arxiv.org/pdf/2412.06239v1)

**Authors**: Mohammed N. Swileh, Shengli Zhang

**Abstract**: Software defined networking (SDN) represents a transformative shift in network architecture by decoupling the control plane from the data plane, enabling centralized and flexible management of network resources. However, this architectural shift introduces significant security challenges, as SDN's centralized control becomes an attractive target for various types of attacks. While current research has yielded valuable insights into attack detection in SDN, critical gaps remain. Addressing challenges in feature selection, broadening the scope beyond DDoS attacks, strengthening attack decisions based on multi flow analysis, and building models capable of detecting unseen attacks that they have not been explicitly trained on are essential steps toward advancing security in SDN. In this paper, we introduce a novel approach that leverages Natural Language Processing (NLP) and the pre trained BERT base model to enhance attack detection in SDN. Our approach transforms network flow data into a format interpretable by language models, allowing BERT to capture intricate patterns and relationships within network traffic. By using Random Forest for feature selection, we optimize model performance and reduce computational overhead, ensuring accurate detection. Attack decisions are made based on several flows, providing stronger and more reliable detection of malicious traffic. Furthermore, our approach is specifically designed to detect previously unseen attacks, offering a solution for identifying threats that the model was not explicitly trained on. To rigorously evaluate our approach, we conducted experiments in two scenarios: one focused on detecting known attacks, achieving 99.96% accuracy, and another on detecting unseen attacks, where our model achieved 99.96% accuracy, demonstrating the robustness of our approach in detecting evolving threats to improve the security of SDN networks.



## **6. Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders**

cs.CL

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2411.07870v4) [paper-pdf](http://arxiv.org/pdf/2411.07870v4)

**Authors**: Xiaofeng Zhu, Jaya Krishna Mandivarapu

**Abstract**: Although people are impressed by the content generation skills of large language models, the use of LLMs, such as ChatGPT, is limited by the domain grounding of the content. The correctness and groundedness of the generated content need to be based on a verified context, such as results from Retrieval-Augmented Generation (RAG). One important issue when adapting LLMs to a customized domain is that the generated responses are often incomplete, or the additions are not verified and may even be hallucinated. Prior studies on hallucination detection have focused on evaluation metrics, which are not easily adaptable to dynamic domains and can be vulnerable to attacks like jail-breaking. In this work, we propose 1) a post-processing algorithm that leverages knowledge triplets in RAG context to correct hallucinations and 2) a dual-decoder model that fuses RAG context to guide the generation process.



## **7. Privacy-Preserving Large Language Models: Mechanisms, Applications, and Future Directions**

cs.CR

**SubmitDate**: 2024-12-09    [abs](http://arxiv.org/abs/2412.06113v1) [paper-pdf](http://arxiv.org/pdf/2412.06113v1)

**Authors**: Guoshenghui Zhao, Eric Song

**Abstract**: The rapid advancement of large language models (LLMs) has revolutionized natural language processing, enabling applications in diverse domains such as healthcare, finance and education. However, the growing reliance on extensive data for training and inference has raised significant privacy concerns, ranging from data leakage to adversarial attacks. This survey comprehensively explores the landscape of privacy-preserving mechanisms tailored for LLMs, including differential privacy, federated learning, cryptographic protocols, and trusted execution environments. We examine their efficacy in addressing key privacy challenges, such as membership inference and model inversion attacks, while balancing trade-offs between privacy and model utility. Furthermore, we analyze privacy-preserving applications of LLMs in privacy-sensitive domains, highlighting successful implementations and inherent limitations. Finally, this survey identifies emerging research directions, emphasizing the need for novel frameworks that integrate privacy by design into the lifecycle of LLMs. By synthesizing state-of-the-art approaches and future trends, this paper provides a foundation for developing robust, privacy-preserving large language models that safeguard sensitive information without compromising performance.



## **8. TrojanRobot: Backdoor Attacks Against LLM-based Embodied Robots in the Physical World**

cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2411.11683v2) [paper-pdf](http://arxiv.org/pdf/2411.11683v2)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.



## **9. Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models**

cs.CR

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2412.05934v1) [paper-pdf](http://arxiv.org/pdf/2412.05934v1)

**Authors**: Ma Teng, Jia Xiaojun, Duan Ranjie, Li Xinfeng, Huang Yihao, Chu Zhixuan, Liu Yang, Ren Wenqi

**Abstract**: With the rapid advancement of multimodal large language models (MLLMs), concerns regarding their security have increasingly captured the attention of both academia and industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective multimodal jailbreak attacks poses unique challenges, especially given the distinct protective measures implemented across various modalities in commercial models. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper, we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which consists of two elements: multimodal risk distribution strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to segment harmful instructions across multiple modalities to effectively circumvent MLLMs' security protection. The heuristic-induced search strategy identifies two types of prompts: the understanding-enhancing prompt, which helps the MLLM reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of affirmative outputs over refusals, enabling a successful jailbreak attack. Extensive experiments demonstrate that this approach effectively uncovers vulnerabilities in MLLMs, achieving an average attack success rate of 90% across seven popular open-source MLLMs and an average attack success rate of around 68% in three popular closed-source MLLMs. Our code will coming soon. Warning: This paper contains offensive and harmful examples, reader discretion is advised.



## **10. Large Language Models Merging for Enhancing the Link Stealing Attack on Graph Neural Networks**

cs.CR

Link Stealing Attacks, Large Language Models, Graph Neural Networks,  Privacy Attacks, Model Merging

**SubmitDate**: 2024-12-08    [abs](http://arxiv.org/abs/2412.05830v1) [paper-pdf](http://arxiv.org/pdf/2412.05830v1)

**Authors**: Faqian Guan, Tianqing Zhu, Wenhan Chang, Wei Ren, Wanlei Zhou

**Abstract**: Graph Neural Networks (GNNs), specifically designed to process the graph data, have achieved remarkable success in various applications. Link stealing attacks on graph data pose a significant privacy threat, as attackers aim to extract sensitive relationships between nodes (entities), potentially leading to academic misconduct, fraudulent transactions, or other malicious activities. Previous studies have primarily focused on single datasets and did not explore cross-dataset attacks, let alone attacks that leverage the combined knowledge of multiple attackers. However, we find that an attacker can combine the data knowledge of multiple attackers to create a more effective attack model, which can be referred to cross-dataset attacks. Moreover, if knowledge can be extracted with the help of Large Language Models (LLMs), the attack capability will be more significant. In this paper, we propose a novel link stealing attack method that takes advantage of cross-dataset and Large Language Models (LLMs). The LLM is applied to process datasets with different data structures in cross-dataset attacks. Each attacker fine-tunes the LLM on their specific dataset to generate a tailored attack model. We then introduce a novel model merging method to integrate the parameters of these attacker-specific models effectively. The result is a merged attack model with superior generalization capabilities, enabling effective attacks not only on the attackers' datasets but also on previously unseen (out-of-domain) datasets. We conducted extensive experiments in four datasets to demonstrate the effectiveness of our method. Additional experiments with three different GNN and LLM architectures further illustrate the generality of our approach.



## **11. Jailbreak Large Vision-Language Models Through Multi-Modal Linkage**

cs.CV

**SubmitDate**: 2024-12-07    [abs](http://arxiv.org/abs/2412.00473v3) [paper-pdf](http://arxiv.org/pdf/2412.00473v3)

**Authors**: Yu Wang, Xiaofei Zhou, Yichen Wang, Geyuan Zhang, Tianxing He

**Abstract**: With the significant advancement of Large Vision-Language Models (VLMs), concerns about their potential misuse and abuse have grown rapidly. Previous studies have highlighted VLMs' vulnerability to jailbreak attacks, where carefully crafted inputs can lead the model to produce content that violates ethical and legal standards. However, existing methods struggle against state-of-the-art VLMs like GPT-4o, due to the over-exposure of harmful content and lack of stealthy malicious guidance. In this work, we propose a novel jailbreak attack framework: Multi-Modal Linkage (MML) Attack. Drawing inspiration from cryptography, MML utilizes an encryption-decryption process across text and image modalities to mitigate over-exposure of malicious information. To align the model's output with malicious intent covertly, MML employs a technique called "evil alignment", framing the attack within a video game production scenario. Comprehensive experiments demonstrate MML's effectiveness. Specifically, MML jailbreaks GPT-4o with attack success rates of 97.80% on SafeBench, 98.81% on MM-SafeBench and 99.07% on HADES-Dataset. Our code is available at https://github.com/wangyu-ovo/MML



## **12. Privacy Risks in Reinforcement Learning for Household Robots**

cs.RO

7 pages, 4 figures, 2 tables

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2306.09273v3) [paper-pdf](http://arxiv.org/pdf/2306.09273v3)

**Authors**: Miao Li, Wenhao Ding, Ding Zhao

**Abstract**: The prominence of embodied Artificial Intelligence (AI), which empowers robots to navigate, perceive, and engage within virtual environments, has attracted significant attention, owing to the remarkable advances in computer vision and large language models. Privacy emerges as a pivotal concern within the realm of embodied AI, as the robot accesses substantial personal information. However, the issue of privacy leakage in embodied AI tasks, particularly concerning reinforcement learning algorithms, has not received adequate consideration in research. This paper aims to address this gap by proposing an attack on the training process of the value-based algorithm and the gradient-based algorithm, utilizing gradient inversion to reconstruct states, actions, and supervisory signals. The choice of using gradients for the attack is motivated by the fact that commonly employed federated learning techniques solely utilize gradients computed based on private user data to optimize models, without storing or transmitting the data to public servers. Nevertheless, these gradients contain sufficient information to potentially expose private data. To validate our approach, we conducted experiments on the AI2THOR simulator and evaluated our algorithm on active perception, a prevalent task in embodied AI. The experimental results demonstrate the effectiveness of our method in successfully reconstructing all information from the data in 120 room layouts. Check our website for videos.



## **13. WAPITI: A Watermark for Finetuned Open-Source LLMs**

cs.CR

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2410.06467v2) [paper-pdf](http://arxiv.org/pdf/2410.06467v2)

**Authors**: Lingjie Chen, Ruizhong Qiu, Siyu Yuan, Zhining Liu, Tianxin Wei, Hyunsik Yoo, Zhichen Zeng, Deqing Yang, Hanghang Tong

**Abstract**: Watermarking of large language models (LLMs) generation embeds an imperceptible statistical pattern within texts, making it algorithmically detectable. Watermarking is a promising method for addressing potential harm and biases from LLMs, as it enables traceability, accountability, and detection of manipulated content, helping to mitigate unintended consequences. However, for open-source models, watermarking faces two major challenges: (i) incompatibility with fine-tuned models, and (ii) vulnerability to fine-tuning attacks. In this work, we propose WAPITI, a new method that transfers watermarking from base models to fine-tuned models through parameter integration. To the best of our knowledge, we propose the first watermark for fine-tuned open-source LLMs that preserves their fine-tuned capabilities. Furthermore, our approach offers an effective defense against fine-tuning attacks. We test our method on various model architectures and watermarking strategies. Results demonstrate that our method can successfully inject watermarks and is highly compatible with fine-tuned models. Additionally, we offer an in-depth analysis of how parameter editing influences the watermark strength and overall capabilities of the resulting models.



## **14. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

cs.CL

8 pages. Submitted to ARR October cycle

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2412.05139v1) [paper-pdf](http://arxiv.org/pdf/2412.05139v1)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, GPTID, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0\%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.



## **15. MultiTrust: A Comprehensive Benchmark Towards Trustworthy Multimodal Large Language Models**

cs.CL

100 pages, 84 figures, 33 tables

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2406.07057v2) [paper-pdf](http://arxiv.org/pdf/2406.07057v2)

**Authors**: Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu

**Abstract**: Despite the superior capabilities of Multimodal Large Language Models (MLLMs) across diverse tasks, they still face significant trustworthiness challenges. Yet, current literature on the assessment of trustworthy MLLMs remains limited, lacking a holistic evaluation to offer thorough insights into future improvements. In this work, we establish MultiTrust, the first comprehensive and unified benchmark on the trustworthiness of MLLMs across five primary aspects: truthfulness, safety, robustness, fairness, and privacy. Our benchmark employs a rigorous evaluation strategy that addresses both multimodal risks and cross-modal impacts, encompassing 32 diverse tasks with self-curated datasets. Extensive experiments with 21 modern MLLMs reveal some previously unexplored trustworthiness issues and risks, highlighting the complexities introduced by the multimodality and underscoring the necessity for advanced methodologies to enhance their reliability. For instance, typical proprietary models still struggle with the perception of visually confusing images and are vulnerable to multimodal jailbreaking and adversarial attacks; MLLMs are more inclined to disclose privacy in text and reveal ideological and cultural biases even when paired with irrelevant images in inference, indicating that the multimodality amplifies the internal risks from base LLMs. Additionally, we release a scalable toolbox for standardized trustworthiness research, aiming to facilitate future advancements in this important field. Code and resources are publicly available at: https://multi-trust.github.io/.



## **16. PropertyGPT: LLM-driven Formal Verification of Smart Contracts through Retrieval-Augmented Property Generation**

cs.SE

Accepted by NDSS Symposium 2025. Please cite the conference version  of this paper, e.g., "Ye Liu, Yue Xue, Daoyuan Wu, Yuqiang Sun, Yi Li,  Miaolei Shi, Yang Liu. PropertyGPT: LLM-driven Formal Verification of Smart  Contracts through Retrieval-Augmented Property Generation. In 32nd Annual  Network and Distributed System Security Symposium (NDSS 2025)."

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2405.02580v2) [paper-pdf](http://arxiv.org/pdf/2405.02580v2)

**Authors**: Ye Liu, Yue Xue, Daoyuan Wu, Yuqiang Sun, Yi Li, Miaolei Shi, Yang Liu

**Abstract**: With recent advances in large language models (LLMs), this paper explores the potential of leveraging state-of-the-art LLMs,such as GPT-4, to transfer existing human-written properties (e.g.,those from Certora auditing reports) and automatically generate customized properties for unknown code. To this end, we embed existing properties into a vector database and retrieve a reference property for LLM-based in-context learning to generate a new property for a given code. While this basic process is relatively straightforward, ensuring that the generated properties are (i) compilable, (ii) appropriate, and (iii) verifiable presents challenges. To address (i), we use the compilation and static analysis feedback as an external oracle to guide LLMs in iteratively revising the generated properties. For (ii), we consider multiple dimensions of similarity to rank the properties and employ a weighted algorithm to identify the top-K properties as the final result. For (iii), we design a dedicated prover to formally verify the correctness of the generated properties. We have implemented these strategies into a novel LLM-based property generation tool called PropertyGPT. Our experiments show that PropertyGPT can generate comprehensive and high-quality properties, achieving an 80% recall compared to the ground truth. It successfully detected 26 CVEs/attack incidents out of 37 tested and also uncovered 12 zero-day vulnerabilities, leading to $8,256 in bug bounty rewards.



## **17. Plentiful Jailbreaks with String Compositions**

cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-12-06    [abs](http://arxiv.org/abs/2411.01084v2) [paper-pdf](http://arxiv.org/pdf/2411.01084v2)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or red-teamers, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary string compositions, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.



## **18. Targeting the Core: A Simple and Effective Method to Attack RAG-based Agents via Direct LLM Manipulation**

cs.AI

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04415v1) [paper-pdf](http://arxiv.org/pdf/2412.04415v1)

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Yasuhiro Yoshida, Victor Bian

**Abstract**: AI agents, powered by large language models (LLMs), have transformed human-computer interactions by enabling seamless, natural, and context-aware communication. While these advancements offer immense utility, they also inherit and amplify inherent safety risks such as bias, fairness, hallucinations, privacy breaches, and a lack of transparency. This paper investigates a critical vulnerability: adversarial attacks targeting the LLM core within AI agents. Specifically, we test the hypothesis that a deceptively simple adversarial prefix, such as \textit{Ignore the document}, can compel LLMs to produce dangerous or unintended outputs by bypassing their contextual safeguards. Through experimentation, we demonstrate a high attack success rate (ASR), revealing the fragility of existing LLM defenses. These findings emphasize the urgent need for robust, multi-layered security measures tailored to mitigate vulnerabilities at the LLM level and within broader agent-based architectures.



## **19. Adversarial Attacks on Large Language Models in Medicine**

cs.AI

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2406.12259v2) [paper-pdf](http://arxiv.org/pdf/2406.12259v2)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.



## **20. Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment**

cs.LG

v2: Updated with changes from peer review rebuttal. v1: Version under  peer review

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2411.02785v2) [paper-pdf](http://arxiv.org/pdf/2411.02785v2)

**Authors**: Jason Vega, Junsheng Huang, Gaokai Zhang, Hangoo Kang, Minjia Zhang, Gagandeep Singh

**Abstract**: Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt. Source code and data: https://github.com/uiuc-focal-lab/stochastic-monkeys/



## **21. Hostility Detection in UK Politics: A Dataset on Online Abuse Targeting MPs**

cs.CL

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04046v1) [paper-pdf](http://arxiv.org/pdf/2412.04046v1)

**Authors**: Mugdha Pandya, Mali Jin, Kalina Bontcheva, Diana Maynard

**Abstract**: Numerous politicians use social media platforms, particularly X, to engage with their constituents. This interaction allows constituents to pose questions and offer feedback but also exposes politicians to a barrage of hostile responses, especially given the anonymity afforded by social media. They are typically targeted in relation to their governmental role, but the comments also tend to attack their personal identity. This can discredit politicians and reduce public trust in the government. It can also incite anger and disrespect, leading to offline harm and violence. While numerous models exist for detecting hostility in general, they lack the specificity required for political contexts. Furthermore, addressing hostility towards politicians demands tailored approaches due to the distinct language and issues inherent to each country (e.g., Brexit for the UK). To bridge this gap, we construct a dataset of 3,320 English tweets spanning a two-year period manually annotated for hostility towards UK MPs. Our dataset also captures the targeted identity characteristics (race, gender, religion, none) in hostile tweets. We perform linguistic and topical analyses to delve into the unique content of the UK political data. Finally, we evaluate the performance of pre-trained language models and large language models on binary hostility detection and multi-class targeted identity type classification tasks. Our study offers valuable data and insights for future research on the prevalence and nature of politics-related hostility specific to the UK.



## **22. R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edge**

eess.SP

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2411.18220v2) [paper-pdf](http://arxiv.org/pdf/2411.18220v2)

**Authors**: Aladin Djuhera, Vlad C. Andrei, Mohsen Pourghasemian, Haris Gacanin, Holger Boche, Walid Saad

**Abstract**: Multi-task large language models (MTLLMs) are important for many applications at the wireless edge, where users demand specialized models to handle multiple tasks efficiently. However, training MTLLMs is complex and exhaustive, particularly when tasks are subject to change. Recently, the concept of model fusion via task vectors has emerged as an efficient approach for combining fine-tuning parameters to produce an MTLLM. In this paper, the problem of enabling edge users to collaboratively craft such MTLMs via tasks vectors is studied, under the assumption of worst-case adversarial attacks. To this end, first the influence of adversarial noise to multi-task model fusion is investigated and a relationship between the so-called weight disentanglement error and the mean squared error (MSE) is derived. Using hypothesis testing, it is directly shown that the MSE increases interference between task vectors, thereby rendering model fusion ineffective. Then, a novel resilient MTLLM fusion (R-MTLLMF) is proposed, which leverages insights about the LLM architecture and fine-tuning process to safeguard task vector aggregation under adversarial noise by realigning the MTLLM. The proposed R-MTLLMF is then compared for both worst-case and ideal transmission scenarios to study the impact of the wireless channel. Extensive model fusion experiments with vision LLMs demonstrate R-MTLLMF's effectiveness, achieving close-to-baseline performance across eight different tasks in ideal noise scenarios and significantly outperforming unprotected model fusion in worst-case scenarios. The results further advocate for additional physical layer protection for a holistic approach to resilience, from both a wireless and LLM perspective.



## **23. AI-based Attacker Models for Enhancing Multi-Stage Cyberattack Simulations in Smart Grids Using Co-Simulation Environments**

cs.CR

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.03979v1) [paper-pdf](http://arxiv.org/pdf/2412.03979v1)

**Authors**: Omer Sen, Christoph Pohl, Immanuel Hacker, Markus Stroot, Andreas Ulbig

**Abstract**: The transition to smart grids has increased the vulnerability of electrical power systems to advanced cyber threats. To safeguard these systems, comprehensive security measures-including preventive, detective, and reactive strategies-are necessary. As part of the critical infrastructure, securing these systems is a major research focus, particularly against cyberattacks. Many methods are developed to detect anomalies and intrusions and assess the damage potential of attacks. However, these methods require large amounts of data, which are often limited or private due to security concerns. We propose a co-simulation framework that employs an autonomous agent to execute modular cyberattacks within a configurable environment, enabling reproducible and adaptable data generation. The impact of virtual attacks is compared to those in a physical lab targeting real smart grids. We also investigate the use of large language models for automating attack generation, though current models on consumer hardware are unreliable. Our approach offers a flexible, versatile source for data generation, aiding in faster prototyping and reducing development resources and time.



## **24. Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization**

cs.LG

31 pages, 45 figures, 7 tables

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2410.12949v2) [paper-pdf](http://arxiv.org/pdf/2410.12949v2)

**Authors**: Phillip Guo, Aaquib Syed, Abhay Sheshadri, Aidan Ewart, Gintare Karolina Dziugaite

**Abstract**: Methods for knowledge editing and unlearning in large language models seek to edit or remove undesirable knowledge or capabilities without compromising general language modeling performance. This work investigates how mechanistic interpretability -- which, in part, aims to identify model components (circuits) associated to specific interpretable mechanisms that make up a model capability -- can improve the precision and effectiveness of editing and unlearning. We find a stark difference in unlearning and edit robustness when training components localized by different methods. We highlight an important distinction between methods that localize components based primarily on preserving outputs, and those finding high level mechanisms with predictable intermediate states. In particular, localizing edits/unlearning to components associated with the lookup-table mechanism for factual recall 1) leads to more robust edits/unlearning across different input/output formats, and 2) resists attempts to relearn the unwanted information, while also reducing unintended side effects compared to baselines, on both a sports facts dataset and the CounterFact dataset across multiple models. We also find that certain localized edits disrupt the latent knowledge in the model more than any other baselines, making unlearning more robust to various attacks.



## **25. WiS Platform: Enhancing Evaluation of LLM-Based Multi-Agent Systems Through Game-Based Analysis**

cs.AI

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03359v1) [paper-pdf](http://arxiv.org/pdf/2412.03359v1)

**Authors**: Chengwei Hu, Jianhui Zheng, Yancheng He, Hangyu Guo, Junguang Jiang, Han Zhu, Kai Sun, Yuning Jiang, Wenbo Su, Bo Zheng

**Abstract**: Recent advancements in autonomous multi-agent systems (MAS) based on large language models (LLMs) have enhanced the application scenarios and improved the capability of LLMs to handle complex tasks. Despite demonstrating effectiveness, existing studies still evidently struggle to evaluate, analysis, and reproducibility of LLM-based MAS. In this paper, to facilitate the research on LLM-based MAS, we introduce an open, scalable, and real-time updated platform for accessing and analyzing the LLM-based MAS based on the games Who is Spy?" (WiS). Our platform is featured with three main worths: (1) a unified model evaluate interface that supports models available on Hugging Face; (2) real-time updated leaderboard for model evaluation; (3) a comprehensive evaluation covering game-winning rates, attacking, defense strategies, and reasoning of LLMs. To rigorously test WiS, we conduct extensive experiments coverage of various open- and closed-source LLMs, we find that different agents exhibit distinct and intriguing behaviors in the game. The experimental results demonstrate the effectiveness and efficiency of our platform in evaluating LLM-based MAS. Our platform and its documentation are publicly available at \url{https://whoisspy.ai/}



## **26. Time-Reversal Provides Unsupervised Feedback to LLMs**

cs.CL

Accepted as a spotlight in NeurIPS 2024

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.02626v2) [paper-pdf](http://arxiv.org/pdf/2412.02626v2)

**Authors**: Yerram Varun, Rahul Madhavan, Sravanti Addepalli, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are typically trained to predict in the forward direction of time. However, recent works have shown that prompting these models to look back and critique their own generations can produce useful feedback. Motivated by this, we explore the question of whether LLMs can be empowered to think (predict and score) backwards to provide unsupervised feedback that complements forward LLMs. Towards this, we introduce Time Reversed Language Models (TRLMs), which can score and generate queries when conditioned on responses, effectively functioning in the reverse direction of time. Further, to effectively infer in the response to query direction, we pre-train and fine-tune a language model (TRLM-Ba) in the reverse token order from scratch. We show empirically (and theoretically in a stylized setting) that time-reversed models can indeed complement forward model predictions when used to score the query given response for re-ranking multiple forward generations. We obtain up to 5\% improvement on the widely used AlpacaEval Leaderboard over the competent baseline of best-of-N re-ranking using self log-perplexity scores. We further show that TRLM scoring outperforms conventional forward scoring of response given query, resulting in significant gains in applications such as citation generation and passage retrieval. We next leverage the generative ability of TRLM to augment or provide unsupervised feedback to input safety filters of LLMs, demonstrating a drastic reduction in false negative rate with negligible impact on false positive rates against several attacks published on the popular JailbreakBench leaderboard.



## **27. Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?**

cs.CL

Accepted at the Safe Generative AI Workshop @ NeurIPS 2024

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03235v1) [paper-pdf](http://arxiv.org/pdf/2412.03235v1)

**Authors**: Sravanti Addepalli, Yerram Varun, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.



## **28. "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks**

cs.CR

This paper has been submitted to Nature Machine Intelligence and  OpenReview preprints. It has 7 pages of text, 3 figures, and 3 tables

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2411.16730v3) [paper-pdf](http://arxiv.org/pdf/2411.16730v3)

**Authors**: Libo Wang

**Abstract**: As the application of large language models continues to expand in various fields, it poses higher challenges to the effectiveness of identifying harmful content generation and guardrail mechanisms. This research aims to evaluate the guardrail effectiveness of GPT-4o, Grok-2 Beta, Llama 3.1 (405B), Gemini 1.5, and Claude 3.5 Sonnet through black-box testing of seemingly ethical multi-step jailbreak prompts. It conducts ethical attacks by designing an identical multi-step prompts that simulates the scenario of "corporate middle managers competing for promotions." The data results show that the guardrails of the above-mentioned LLMs were bypassed and the content of verbal attacks was generated. Claude 3.5 Sonnet's resistance to multi-step jailbreak prompts is more obvious. To ensure objectivity, the experimental process, black box test code, and enhanced guardrail code are uploaded to the GitHub repository: https://github.com/brucewang123456789/GeniusTrail.git.



## **29. Backdoor Attacks and Countermeasures in Natural Language Processing Models: A Comprehensive Security Review**

cs.CR

21 pages, 3 figures

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2309.06055v5) [paper-pdf](http://arxiv.org/pdf/2309.06055v5)

**Authors**: Pengzhou Cheng, Zongru Wu, Wei Du, Haodong Zhao, Wei Lu, Gongshen Liu

**Abstract**: Language Models (LMs) are becoming increasingly popular in real-world applications. Outsourcing model training and data hosting to third-party platforms has become a standard method for reducing costs. In such a situation, the attacker can manipulate the training process or data to inject a backdoor into models. Backdoor attacks are a serious threat where malicious behavior is activated when triggers are present, otherwise, the model operates normally.   However, there is still no systematic and comprehensive review of LMs from the attacker's capabilities and purposes on different backdoor attack surfaces. Moreover, there is a shortage of analysis and comparison of the diverse emerging backdoor countermeasures. Therefore, this work aims to provide the NLP community with a timely review of backdoor attacks and countermeasures. According to the attackers' capability and affected stage of the LMs, the attack surfaces are formalized into four categorizations: attacking the pre-trained model with fine-tuning (APMF) or parameter-efficient fine-tuning (APMP), attacking the final model with training (AFMT), and attacking Large Language Models (ALLM). Thus, attacks under each categorization are combed. The countermeasures are categorized into two general classes: sample inspection and model inspection. Thus, we review countermeasures and analyze their advantages and disadvantages. Also, we summarize the benchmark datasets and provide comparable evaluations for representative attacks and defenses. Drawing the insights from the review, we point out the crucial areas for future research on the backdoor, especially soliciting more efficient and practical countermeasures.



## **30. Unleashing GHOST: An LLM-Powered Framework for Automated Hardware Trojan Design**

cs.CR

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02816v1) [paper-pdf](http://arxiv.org/pdf/2412.02816v1)

**Authors**: Md Omar Faruque, Peter Jamieson, Ahmad Patooghy, Abdel-Hameed A. Badawy

**Abstract**: Traditionally, inserting realistic Hardware Trojans (HTs) into complex hardware systems has been a time-consuming and manual process, requiring comprehensive knowledge of the design and navigating intricate Hardware Description Language (HDL) codebases. Machine Learning (ML)-based approaches have attempted to automate this process but often face challenges such as the need for extensive training data, long learning times, and limited generalizability across diverse hardware design landscapes. This paper addresses these challenges by proposing GHOST (Generator for Hardware-Oriented Stealthy Trojans), an automated attack framework that leverages Large Language Models (LLMs) for rapid HT generation and insertion. Our study evaluates three state-of-the-art LLMs - GPT-4, Gemini-1.5-pro, and Llama-3-70B - across three hardware designs: SRAM, AES, and UART. According to our evaluations, GPT-4 demonstrates superior performance, with 88.88% of HT insertion attempts successfully generating functional and synthesizable HTs. This study also highlights the security risks posed by LLM-generated HTs, showing that 100% of GHOST-generated synthesizable HTs evaded detection by an ML-based HT detection tool. These results underscore the urgent need for advanced detection and prevention mechanisms in hardware security to address the emerging threat of LLM-generated HTs. The GHOST HT benchmarks are available at: https://github.com/HSTRG1/GHOSTbenchmarks.git



## **31. Gracefully Filtering Backdoor Samples for Generative Large Language Models without Retraining**

cs.CL

Accepted at COLING 2025

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02454v1) [paper-pdf](http://arxiv.org/pdf/2412.02454v1)

**Authors**: Zongru Wu, Pengzhou Cheng, Lingyong Fang, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Backdoor attacks remain significant security threats to generative large language models (LLMs). Since generative LLMs output sequences of high-dimensional token logits instead of low-dimensional classification logits, most existing backdoor defense methods designed for discriminative models like BERT are ineffective for generative LLMs. Inspired by the observed differences in learning behavior between backdoor and clean mapping in the frequency space, we transform gradients of each training sample, directly influencing parameter updates, into the frequency space. Our findings reveal a distinct separation between the gradients of backdoor and clean samples in the frequency space. Based on this phenomenon, we propose Gradient Clustering in the Frequency Space for Backdoor Sample Filtering (GraCeFul), which leverages sample-wise gradients in the frequency space to effectively identify backdoor samples without requiring retraining LLMs. Experimental results show that GraCeFul outperforms baselines significantly. Notably, GraCeFul exhibits remarkable computational efficiency, achieving nearly 100% recall and F1 scores in identifying backdoor samples, reducing the average success rate of various backdoor attacks to 0% with negligible drops in clean accuracy across multiple free-style question answering datasets. Additionally, GraCeFul generalizes to Llama-2 and Vicuna. The codes are publicly available at https://github.com/ZrW00/GraceFul.



## **32. Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2409.18169v5) [paper-pdf](http://arxiv.org/pdf/2409.18169v5)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent research demonstrates that the nascent fine-tuning-as-a-service business model exposes serious safety concerns -- fine-tuning over a few harmful data uploaded by the users can compromise the safety alignment of the model. The attack, known as harmful fine-tuning attack, has raised a broad research interest among the community. However, as the attack is still new, \textbf{we observe that there are general misunderstandings within the research community.} To clear up concern, this paper provide a comprehensive overview to three aspects of harmful fine-tuning: attacks setting, defense design and evaluation methodology. Specifically, we first present the threat model of the problem, and introduce the harmful fine-tuning attack and its variants. Then we systematically survey the existing literature on attacks/defenses/mechanical analysis of the problem. Finally, we introduce the evaluation methodology and outline future research directions that might contribute to the development of the field. Additionally, we present a list of questions of interest, which might be useful to refer to when reviewers in the peer review process question the realism of the experiment/attack/defense setting. A curated list of relevant papers is maintained and made accessible at: https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers.



## **33. Trust & Safety of LLMs and LLMs in Trust & Safety**

cs.AI

11 pages

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02113v1) [paper-pdf](http://arxiv.org/pdf/2412.02113v1)

**Authors**: Doohee You, Dan Chon

**Abstract**: In recent years, Large Language Models (LLMs) have garnered considerable attention for their remarkable abilities in natural language processing tasks. However, their widespread adoption has raised concerns pertaining to trust and safety. This systematic review investigates the current research landscape on trust and safety in LLMs, with a particular focus on the novel application of LLMs within the field of Trust and Safety itself. We delve into the complexities of utilizing LLMs in domains where maintaining trust and safety is paramount, offering a consolidated perspective on this emerging trend.\   By synthesizing findings from various studies, we identify key challenges and potential solutions, aiming to benefit researchers and practitioners seeking to understand the nuanced interplay between LLMs and Trust and Safety.   This review provides insights on best practices for using LLMs in Trust and Safety, and explores emerging risks such as prompt injection and jailbreak attacks. Ultimately, this study contributes to a deeper understanding of how LLMs can be effectively and responsibly utilized to enhance trust and safety in the digital realm.



## **34. Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis**

cs.CL

Accepted by EMNLP 2024 Main

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2406.10794v3) [paper-pdf](http://arxiv.org/pdf/2406.10794v3)

**Authors**: Yuping Lin, Pengfei He, Han Xu, Yue Xing, Makoto Yamada, Hui Liu, Jiliang Tang

**Abstract**: Large language models (LLMs) are susceptible to a type of attack known as jailbreaking, which misleads LLMs to output harmful contents. Although there are diverse jailbreak attack strategies, there is no unified understanding on why some methods succeed and others fail. This paper explores the behavior of harmful and harmless prompts in the LLM's representation space to investigate the intrinsic properties of successful jailbreak attacks. We hypothesize that successful attacks share some similar properties: They are effective in moving the representation of the harmful prompt towards the direction to the harmless prompts. We leverage hidden representations into the objective of existing jailbreak attacks to move the attacks along the acceptance direction, and conduct experiments to validate the above hypothesis using the proposed objective. We hope this study provides new insights into understanding how LLMs understand harmfulness information.



## **35. Improved Large Language Model Jailbreak Detection via Pretrained Embeddings**

cs.CR

Submitted to AICS 2025: https://aics.site

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01547v1) [paper-pdf](http://arxiv.org/pdf/2412.01547v1)

**Authors**: Erick Galinkin, Martin Sablotny

**Abstract**: The adoption of large language models (LLMs) in many applications, from customer service chat bots and software development assistants to more capable agentic systems necessitates research into how to secure these systems. Attacks like prompt injection and jailbreaking attempt to elicit responses and actions from these models that are not compliant with the safety, privacy, or content policies of organizations using the model in their application. In order to counter abuse of LLMs for generating potentially harmful replies or taking undesirable actions, LLM owners must apply safeguards during training and integrate additional tools to block the LLM from generating text that abuses the model. Jailbreaking prompts play a vital role in convincing an LLM to generate potentially harmful content, making it important to identify jailbreaking attempts to block any further steps. In this work, we propose a novel approach to detect jailbreak prompts based on pairing text embeddings well-suited for retrieval with traditional machine learning classification algorithms. Our approach outperforms all publicly available methods from open source LLM security applications.



## **36. LUMIA: Linear probing for Unimodal and MultiModal Membership Inference Attacks leveraging internal LLM states**

cs.CR

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2411.19876v2) [paper-pdf](http://arxiv.org/pdf/2411.19876v2)

**Authors**: Luis Ibanez-Lissen, Lorena Gonzalez-Manzano, Jose Maria de Fuentes, Nicolas Anciaux, Joaquin Garcia-Alfaro

**Abstract**: Large Language Models (LLMs) are increasingly used in a variety of applications, but concerns around membership inference have grown in parallel. Previous efforts focus on black-to-grey-box models, thus neglecting the potential benefit from internal LLM information. To address this, we propose the use of Linear Probes (LPs) as a method to detect Membership Inference Attacks (MIAs) by examining internal activations of LLMs. Our approach, dubbed LUMIA, applies LPs layer-by-layer to get fine-grained data on the model inner workings. We test this method across several model architectures, sizes and datasets, including unimodal and multimodal tasks. In unimodal MIA, LUMIA achieves an average gain of 15.71 % in Area Under the Curve (AUC) over previous techniques. Remarkably, LUMIA reaches AUC>60% in 65.33% of cases -- an increment of 46.80% against the state of the art. Furthermore, our approach reveals key insights, such as the model layers where MIAs are most detectable. In multimodal models, LPs indicate that visual inputs can significantly contribute to detect MIAs -- AUC>60% is reached in 85.90% of experiments.



## **37. Recent Advances in Attack and Defense Approaches of Large Language Models**

cs.CR

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2409.03274v3) [paper-pdf](http://arxiv.org/pdf/2409.03274v3)

**Authors**: Jing Cui, Yishi Xu, Zhewei Huang, Shuchang Zhou, Jianbin Jiao, Junge Zhang

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence and machine learning through their advanced text processing and generating capabilities. However, their widespread deployment has raised significant safety and reliability concerns. Established vulnerabilities in deep neural networks, coupled with emerging threat models, may compromise security evaluations and create a false sense of security. Given the extensive research in the field of LLM security, we believe that summarizing the current state of affairs will help the research community better understand the present landscape and inform future developments. This paper reviews current research on LLM vulnerabilities and threats, and evaluates the effectiveness of contemporary defense mechanisms. We analyze recent studies on attack vectors and model weaknesses, providing insights into attack mechanisms and the evolving threat landscape. We also examine current defense strategies, highlighting their strengths and limitations. By contrasting advancements in attack and defense methodologies, we identify research gaps and propose future directions to enhance LLM security. Our goal is to advance the understanding of LLM safety challenges and guide the development of more robust security measures.



## **38. BDefects4NN: A Backdoor Defect Database for Controlled Localization Studies in Neural Networks**

cs.SE

11 pages, accepted by ICSE 2025

**SubmitDate**: 2024-12-01    [abs](http://arxiv.org/abs/2412.00746v1) [paper-pdf](http://arxiv.org/pdf/2412.00746v1)

**Authors**: Yisong Xiao, Aishan Liu, Xinwei Zhang, Tianyuan Zhang, Tianlin Li, Siyuan Liang, Xianglong Liu, Yang Liu, Dacheng Tao

**Abstract**: Pre-trained large deep learning models are now serving as the dominant component for downstream middleware users and have revolutionized the learning paradigm, replacing the traditional approach of training from scratch locally. To reduce development costs, developers often integrate third-party pre-trained deep neural networks (DNNs) into their intelligent software systems. However, utilizing untrusted DNNs presents significant security risks, as these models may contain intentional backdoor defects resulting from the black-box training process. These backdoor defects can be activated by hidden triggers, allowing attackers to maliciously control the model and compromise the overall reliability of the intelligent software. To ensure the safe adoption of DNNs in critical software systems, it is crucial to establish a backdoor defect database for localization studies. This paper addresses this research gap by introducing BDefects4NN, the first backdoor defect database, which provides labeled backdoor-defected DNNs at the neuron granularity and enables controlled localization studies of defect root causes. In BDefects4NN, we define three defect injection rules and employ four representative backdoor attacks across four popular network architectures and three widely adopted datasets, yielding a comprehensive database of 1,654 backdoor-defected DNNs with four defect quantities and varying infected neurons. Based on BDefects4NN, we conduct extensive experiments on evaluating six fault localization criteria and two defect repair techniques, which show limited effectiveness for backdoor defects. Additionally, we investigate backdoor-defected models in practical scenarios, specifically in lane detection for autonomous driving and large language models (LLMs), revealing potential threats and highlighting current limitations in precise defect localization.



## **39. Evaluating Large Language Models' Capability to Launch Fully Automated Spear Phishing Campaigns: Validated on Human Subjects**

cs.CR

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2412.00586v1) [paper-pdf](http://arxiv.org/pdf/2412.00586v1)

**Authors**: Fred Heiding, Simon Lermen, Andrew Kao, Bruce Schneier, Arun Vishwanath

**Abstract**: In this paper, we evaluate the capability of large language models to conduct personalized phishing attacks and compare their performance with human experts and AI models from last year. We include four email groups with a combined total of 101 participants: A control group of arbitrary phishing emails, which received a click-through rate (recipient pressed a link in the email) of 12%, emails generated by human experts (54% click-through), fully AI-automated emails 54% (click-through), and AI emails utilizing a human-in-the-loop (56% click-through). Thus, the AI-automated attacks performed on par with human experts and 350% better than the control group. The results are a significant improvement from similar studies conducted last year, highlighting the increased deceptive capabilities of AI models. Our AI-automated emails were sent using a custom-built tool that automates the entire spear phishing process, including information gathering and creating personalized vulnerability profiles for each target. The AI-gathered information was accurate and useful in 88% of cases and only produced inaccurate profiles for 4% of the participants. We also use language models to detect the intention of emails. Claude 3.5 Sonnet scored well above 90% with low false-positive rates and detected several seemingly benign emails that passed human detection. Lastly, we analyze the economics of phishing, highlighting how AI enables attackers to target more individuals at lower cost and increase profitability by up to 50 times for larger audiences.



## **40. Uncovering Safety Risks of Large Language Models through Concept Activation Vector**

cs.CL

10 pages, accepted at NeurIPS 2024

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2404.12038v5) [paper-pdf](http://arxiv.org/pdf/2404.12038v5)

**Authors**: Zhihao Xu, Ruixuan Huang, Changyu Chen, Xiting Wang

**Abstract**: Despite careful safety alignment, current large language models (LLMs) remain vulnerable to various attacks. To further unveil the safety risks of LLMs, we introduce a Safety Concept Activation Vector (SCAV) framework, which effectively guides the attacks by accurately interpreting LLMs' safety mechanisms. We then develop an SCAV-guided attack method that can generate both attack prompts and embedding-level attacks with automatically selected perturbation hyperparameters. Both automatic and human evaluations demonstrate that our attack method significantly improves the attack success rate and response quality while requiring less training data. Additionally, we find that our generated attack prompts may be transferable to GPT-4, and the embedding-level attacks may also be transferred to other white-box LLMs whose parameters are known. Our experiments further uncover the safety risks present in current LLMs. For example, in our evaluation of seven open-source LLMs, we observe an average attack success rate of 99.14%, based on the classic keyword-matching criterion. Finally, we provide insights into the safety mechanism of LLMs. The code is available at https://github.com/SproutNan/AI-Safety_SCAV.



## **41. Safety Alignment Backfires: Preventing the Re-emergence of Suppressed Concepts in Fine-tuned Text-to-Image Diffusion Models**

cs.AI

20 pages, 18 figures

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2412.00357v1) [paper-pdf](http://arxiv.org/pdf/2412.00357v1)

**Authors**: Sanghyun Kim, Moonseok Choi, Jinwoo Shin, Juho Lee

**Abstract**: Fine-tuning text-to-image diffusion models is widely used for personalization and adaptation for new domains. In this paper, we identify a critical vulnerability of fine-tuning: safety alignment methods designed to filter harmful content (e.g., nudity) can break down during fine-tuning, allowing previously suppressed content to resurface, even when using benign datasets. While this "fine-tuning jailbreaking" issue is known in large language models, it remains largely unexplored in text-to-image diffusion models. Our investigation reveals that standard fine-tuning can inadvertently undo safety measures, causing models to relearn harmful concepts that were previously removed and even exacerbate harmful behaviors. To address this issue, we present a novel but immediate solution called Modular LoRA, which involves training Safety Low-Rank Adaptation (LoRA) modules separately from Fine-Tuning LoRA components and merging them during inference. This method effectively prevents the re-learning of harmful content without compromising the model's performance on new tasks. Our experiments demonstrate that Modular LoRA outperforms traditional fine-tuning methods in maintaining safety alignment, offering a practical approach for enhancing the security of text-to-image diffusion models against potential attacks.



## **42. When LLMs Go Online: The Emerging Threat of Web-Enabled LLMs**

cs.CR

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2410.14569v2) [paper-pdf](http://arxiv.org/pdf/2410.14569v2)

**Authors**: Hanna Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin, Kimin Lee

**Abstract**: Recent advancements in Large Language Models (LLMs) have established them as agentic systems capable of planning and interacting with various tools. These LLM agents are often paired with web-based tools, enabling access to diverse sources and real-time information. Although these advancements offer significant benefits across various applications, they also increase the risk of malicious use, particularly in cyberattacks involving personal information. In this work, we investigate the risks associated with misuse of LLM agents in cyberattacks involving personal data. Specifically, we aim to understand: 1) how potent LLM agents can be when directed to conduct cyberattacks, 2) how cyberattacks are enhanced by web-based tools, and 3) how affordable and easy it becomes to launch cyberattacks using LLM agents. We examine three attack scenarios: the collection of Personally Identifiable Information (PII), the generation of impersonation posts, and the creation of spear-phishing emails. Our experiments reveal the effectiveness of LLM agents in these attacks: LLM agents achieved a precision of up to 95.9% in collecting PII, up to 93.9% of impersonation posts created by LLM agents were evaluated as authentic, and the click rate for links in spear phishing emails created by LLM agents reached up to 46.67%. Additionally, our findings underscore the limitations of existing safeguards in contemporary commercial LLMs, emphasizing the urgent need for more robust security measures to prevent the misuse of LLM agents.



## **43. Ensemble Watermarks for Large Language Models**

cs.CL

9 pages in the main body. Code is available at  http://github.com/CommodoreEU/master-generation. arXiv admin note:  substantial text overlap with arXiv:2405.08400

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2411.19563v1) [paper-pdf](http://arxiv.org/pdf/2411.19563v1)

**Authors**: Georg Niess, Roman Kern

**Abstract**: The rapid advancement of large language models (LLMs) has made it increasingly difficult to distinguish between text written by humans and machines. While watermarks already exist for LLMs, they often lack flexibility, and struggle with attacks such as paraphrasing. To address these issues, we propose a multi-feature method for generating watermarks that combines multiple distinct watermark features into an ensemble watermark. Concretely, we combine acrostica and sensorimotor norms with the established red-green watermark to achieve a 98% detection rate. After a paraphrasing attack the performance remains high with 95% detection rate. The red-green feature alone as baseline achieves a detection rate of 49%. The evaluation of all feature combinations reveals that the ensemble of all three consistently has the highest detection rate across several LLMs and watermark strength settings. Due to the flexibility of combining features in the ensemble, various requirements and trade-offs can be addressed. Additionally, for all ensemble configurations the same detection function can be used without adaptations. This method is particularly of interest to facilitate accountability and prevent societal harm.



## **44. InputSnatch: Stealing Input in LLM Services via Timing Side-Channel Attacks**

cs.CR

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2411.18191v2) [paper-pdf](http://arxiv.org/pdf/2411.18191v2)

**Authors**: Xinyao Zheng, Husheng Han, Shangyi Shi, Qiyan Fang, Zidong Du, Xing Hu, Qi Guo

**Abstract**: Large language models (LLMs) possess extensive knowledge and question-answering capabilities, having been widely deployed in privacy-sensitive domains like finance and medical consultation. During LLM inferences, cache-sharing methods are commonly employed to enhance efficiency by reusing cached states or responses for the same or similar inference requests. However, we identify that these cache mechanisms pose a risk of private input leakage, as the caching can result in observable variations in response times, making them a strong candidate for a timing-based attack hint.   In this study, we propose a novel timing-based side-channel attack to execute input theft in LLMs inference. The cache-based attack faces the challenge of constructing candidate inputs in a large search space to hit and steal cached user queries. To address these challenges, we propose two primary components. The input constructor employs machine learning techniques and LLM-based approaches for vocabulary correlation learning while implementing optimized search mechanisms for generalized input construction. The time analyzer implements statistical time fitting with outlier elimination to identify cache hit patterns, continuously providing feedback to refine the constructor's search strategy. We conduct experiments across two cache mechanisms and the results demonstrate that our approach consistently attains high attack success rates in various applications. Our work highlights the security vulnerabilities associated with performance optimizations, underscoring the necessity of prioritizing privacy and security alongside enhancements in LLM inference.



## **45. RePD: Defending Jailbreak Attack through a Retrieval-based Prompt Decomposition Process**

cs.CR

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2410.08660v3) [paper-pdf](http://arxiv.org/pdf/2410.08660v3)

**Authors**: Peiran Wang, Xiaogeng Liu, Chaowei Xiao

**Abstract**: In this study, we introduce RePD, an innovative attack Retrieval-based Prompt Decomposition framework designed to mitigate the risk of jailbreak attacks on large language models (LLMs). Despite rigorous pretraining and finetuning focused on ethical alignment, LLMs are still susceptible to jailbreak exploits. RePD operates on a one-shot learning model, wherein it accesses a database of pre-collected jailbreak prompt templates to identify and decompose harmful inquiries embedded within user prompts. This process involves integrating the decomposition of the jailbreak prompt into the user's original query into a one-shot learning example to effectively teach the LLM to discern and separate malicious components. Consequently, the LLM is equipped to first neutralize any potentially harmful elements before addressing the user's prompt in a manner that aligns with its ethical guidelines. RePD is versatile and compatible with a variety of open-source LLMs acting as agents. Through comprehensive experimentation with both harmful and benign prompts, we have demonstrated the efficacy of our proposed RePD in enhancing the resilience of LLMs against jailbreak attacks, without compromising their performance in responding to typical user requests.



## **46. Confidential Prompting: Protecting User Prompts from Cloud LLM Providers**

cs.CR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2409.19134v2) [paper-pdf](http://arxiv.org/pdf/2409.19134v2)

**Authors**: In Gim, Caihua Li, Lin Zhong

**Abstract**: Our work tackles the challenge of securing user inputs in cloud-hosted large language model (LLM) serving while ensuring output invariance, model confidentiality, and compute efficiency. We introduce secure multi-party decoding (SMD), which leverages confidential computing to confine user prompts to a trusted execution environment (TEE), namely a confidential virtual machine (CVM), while allowing service providers to generate tokens efficiently. We also introduce a novel cryptographic method, prompt obfuscation (PO), to ensure robustness against reconstruction attacks on SMD. We demonstrate that our approach preserves both prompt confidentiality and LLM serving efficiency. Our solution can enable privacy-preserving cloud LLM serving that handles sensitive prompts, such as clinical records, financial data, and personal information.



## **47. Memorization of Named Entities in Fine-tuned BERT Models**

cs.CL

published at CD-MAKE 2023

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2212.03749v3) [paper-pdf](http://arxiv.org/pdf/2212.03749v3)

**Authors**: Andor Diera, Nicolas Lell, Aygul Garifullina, Ansgar Scherp

**Abstract**: Privacy preserving deep learning is an emerging field in machine learning that aims to mitigate the privacy risks in the use of deep neural networks. One such risk is training data extraction from language models that have been trained on datasets, which contain personal and privacy sensitive information. In our study, we investigate the extent of named entity memorization in fine-tuned BERT models. We use single-label text classification as representative downstream task and employ three different fine-tuning setups in our experiments, including one with Differential Privacy (DP). We create a large number of text samples from the fine-tuned BERT models utilizing a custom sequential sampling strategy with two prompting strategies. We search in these samples for named entities and check if they are also present in the fine-tuning datasets. We experiment with two benchmark datasets in the domains of emails and blogs. We show that the application of DP has a detrimental effect on the text generation capabilities of BERT. Furthermore, we show that a fine-tuned BERT does not generate more named entities specific to the fine-tuning dataset than a BERT model that is pre-trained only. This suggests that BERT is unlikely to emit personal or privacy sensitive named entities. Overall, our results are important to understand to what extent BERT-based services are prone to training data extraction attacks.



## **48. On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks**

cs.CR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2407.04794v2) [paper-pdf](http://arxiv.org/pdf/2407.04794v2)

**Authors**: Zesen Liu, Tianshuo Cong, Xinlei He, Qi Li

**Abstract**: Large Language Models (LLMs) excel in various applications, including text generation and complex tasks. However, the misuse of LLMs raises concerns about the authenticity and ethical implications of the content they produce, such as deepfake news, academic fraud, and copyright infringement. Watermarking techniques, which embed identifiable markers in machine-generated text, offer a promising solution to these issues by allowing for content verification and origin tracing. Unfortunately, the robustness of current LLM watermarking schemes under potential watermark removal attacks has not been comprehensively explored.   In this paper, to fill this gap, we first systematically comb the mainstream watermarking schemes and removal attacks on machine-generated texts, and then we categorize them into pre-text (before text generation) and post-text (after text generation) classes so that we can conduct diversified analyses. In our experiments, we evaluate eight watermarks (five pre-text, three post-text) and twelve attacks (two pre-text, ten post-text) across 87 scenarios. Evaluation results indicate that (1) KGW and Exponential watermarks offer high text quality and watermark retention but remain vulnerable to most attacks; (2) Post-text attacks are found to be more efficient and practical than pre-text attacks; (3) Pre-text watermarks are generally more imperceptible, as they do not alter text fluency, unlike post-text watermarks; (4) Additionally, combined attack methods can significantly increase effectiveness, highlighting the need for more robust watermarking solutions. Our study underscores the vulnerabilities of current techniques and the necessity for developing more resilient schemes.



## **49. Assessing biomedical knowledge robustness in large language models by query-efficient sampling attacks**

cs.CL

31 pages incl. appendix, accepted by TMLR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2402.10527v3) [paper-pdf](http://arxiv.org/pdf/2402.10527v3)

**Authors**: R. Patrick Xian, Alex J. Lee, Satvik Lolla, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. Understanding model vulnerabilities in high-stakes and knowledge-intensive tasks is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples (i.e. adversarial entities) in natural language processing tasks raises questions about their potential impact on the knowledge robustness of pre-trained and finetuned LLMs in high-stakes and specialized domains. We examined the use of type-consistent entity substitution as a template for collecting adversarial entities for billion-parameter LLMs with biomedical knowledge. To this end, we developed an embedding-space attack based on powerscaled distance-weighted sampling to assess the robustness of their biomedical knowledge with a low query budget and controllable coverage. Our method has favorable query efficiency and scaling over alternative approaches based on random sampling and blackbox gradient-guided search, which we demonstrated for adversarial distractor generation in biomedical question answering. Subsequent failure mode analysis uncovered two regimes of adversarial entities on the attack surface with distinct characteristics and we showed that entity substitution attacks can manipulate token-wise Shapley value explanations, which become deceptive in this setting. Our approach complements standard evaluations for high-capacity models and the results highlight the brittleness of domain knowledge in LLMs.



## **50. Knowledge Database or Poison Base? Detecting RAG Poisoning Attack through LLM Activations**

cs.CR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2411.18948v1) [paper-pdf](http://arxiv.org/pdf/2411.18948v1)

**Authors**: Xue Tan, Hao Luan, Mingyu Luo, Xiaoyan Sun, Ping Chen, Jun Dai

**Abstract**: As Large Language Models (LLMs) are progressively deployed across diverse fields and real-world applications, ensuring the security and robustness of LLMs has become ever more critical. Retrieval-Augmented Generation (RAG) is a cutting-edge approach designed to address the limitations of large language models (LLMs). By retrieving information from the relevant knowledge database, RAG enriches the input to LLMs, enabling them to produce responses that are more accurate and contextually appropriate. It is worth noting that the knowledge database, being sourced from publicly available channels such as Wikipedia, inevitably introduces a new attack surface. RAG poisoning involves injecting malicious texts into the knowledge database, ultimately leading to the generation of the attacker's target response (also called poisoned response). However, there are currently limited methods available for detecting such poisoning attacks. We aim to bridge the gap in this work. Particularly, we introduce RevPRAG, a flexible and automated detection pipeline that leverages the activations of LLMs for poisoned response detection. Our investigation uncovers distinct patterns in LLMs' activations when generating correct responses versus poisoned responses. Our results on multiple benchmark datasets and RAG architectures show our approach could achieve 98% true positive rate, while maintaining false positive rates close to 1%. We also evaluate recent backdoor detection methods specifically designed for LLMs and applicable for identifying poisoned responses in RAG. The results demonstrate that our approach significantly surpasses them.



