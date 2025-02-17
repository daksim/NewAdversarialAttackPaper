# Latest Large Language Model Attack Papers
**update at 2025-02-17 10:29:47**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. SequentialBreak: Large Language Models Can be Fooled by Embedding Jailbreak Prompts into Sequential Prompt Chains**

cs.CR

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2411.06426v2) [paper-pdf](http://arxiv.org/pdf/2411.06426v2)

**Authors**: Bijoy Ahmed Saiem, MD Sadik Hossain Shanto, Rakib Ahsan, Md Rafi ur Rashid

**Abstract**: As the integration of the Large Language Models (LLMs) into various applications increases, so does their susceptibility to misuse, raising significant security concerns. Numerous jailbreak attacks have been proposed to assess the security defense of LLMs. Current jailbreak attacks mainly rely on scenario camouflage, prompt obfuscation, prompt optimization, and prompt iterative optimization to conceal malicious prompts. In particular, sequential prompt chains in a single query can lead LLMs to focus on certain prompts while ignoring others, facilitating context manipulation. This paper introduces SequentialBreak, a novel jailbreak attack that exploits this vulnerability. We discuss several scenarios, not limited to examples like Question Bank, Dialog Completion, and Game Environment, where the harmful prompt is embedded within benign ones that can fool LLMs into generating harmful responses. The distinct narrative structures of these scenarios show that SequentialBreak is flexible enough to adapt to various prompt formats beyond those discussed. Extensive experiments demonstrate that SequentialBreak uses only a single query to achieve a substantial gain of attack success rate over existing baselines against both open-source and closed-source models. Through our research, we highlight the urgent need for more robust and resilient safeguards to enhance LLM security and prevent potential misuse. All the result files and website associated with this research are available in this GitHub repository: https://anonymous.4open.science/r/JailBreakAttack-4F3B/.



## **2. Translating Common Security Assertions Across Processor Designs: A RISC-V Case Study**

cs.CR

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.10194v1) [paper-pdf](http://arxiv.org/pdf/2502.10194v1)

**Authors**: Sharjeel Imtiaz, Uljana Reinsalu, Tara Ghasempouri

**Abstract**: RISC-V is gaining popularity for its adaptability and cost-effectiveness in processor design. With the increasing adoption of RISC-V, the importance of implementing robust security verification has grown significantly. In the state of the art, various approaches have been developed to strengthen the security verification process. Among these methods, assertion-based security verification has proven to be a promising approach for ensuring that security features are effectively met. To this end, some approaches manually define security assertions for processor designs; however, these manual methods require significant time, cost, and human expertise. Consequently, recent approaches focus on translating pre-defined security assertions from one design to another. Nonetheless, these methods are not primarily centered on processor security, particularly RISC-V. Furthermore, many of these approaches have not been validated against real-world attacks, such as hardware Trojans. In this work, we introduce a methodology for translating security assertions across processors with different architectures, using RISC-V as a case study. Our approach reduces time and cost compared to developing security assertions manually from the outset. Our methodology was applied to five critical security modules with assertion translation achieving nearly 100% success across all modules. These results validate the efficacy of our approach and highlight its potential for enhancing security verification in modern processor designs. The effectiveness of the translated assertions was rigorously tested against hardware Trojans defined by large language models (LLMs), demonstrating their reliability in detecting security breaches.



## **3. What You See Is Not Always What You Get: An Empirical Study of Code Comprehension by Large Language Models**

cs.SE

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2412.08098v2) [paper-pdf](http://arxiv.org/pdf/2412.08098v2)

**Authors**: Bangshuo Zhu, Jiawen Wen, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering tasks, including code generation and comprehension. While LLMs have shown significant potential in assisting with coding, it is perceived that LLMs are vulnerable to adversarial attacks. In this paper, we investigate the vulnerability of LLMs to imperceptible attacks, where hidden character manipulation in source code misleads LLMs' behaviour while remaining undetectable to human reviewers. We devise these attacks into four distinct categories and analyse their impacts on code analysis and comprehension tasks. These four types of imperceptible coding character attacks include coding reordering, invisible coding characters, code deletions, and code homoglyphs. To comprehensively benchmark the robustness of current LLMs solutions against the attacks, we present a systematic experimental evaluation on multiple state-of-the-art LLMs. Our experimental design introduces two key performance metrics, namely model confidence using log probabilities of response, and the response correctness. A set of controlled experiments are conducted using a large-scale perturbed and unperturbed code snippets as the primary prompt input. Our findings confirm the susceptibility of LLMs to imperceptible coding character attacks, while different LLMs present different negative correlations between perturbation magnitude and performance. These results highlight the urgent need for robust LLMs capable of manoeuvring behaviours under imperceptible adversarial conditions. We anticipate this work provides valuable insights for enhancing the security and trustworthiness of LLMs in software engineering applications.



## **4. Detecting Phishing Sites Using ChatGPT**

cs.CR

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2306.05816v3) [paper-pdf](http://arxiv.org/pdf/2306.05816v3)

**Authors**: Takashi Koide, Naoki Fukushi, Hiroki Nakano, Daiki Chiba

**Abstract**: The emergence of Large Language Models (LLMs), including ChatGPT, is having a significant impact on a wide range of fields. While LLMs have been extensively researched for tasks such as code generation and text synthesis, their application in detecting malicious web content, particularly phishing sites, has been largely unexplored. To combat the rising tide of cyber attacks due to the misuse of LLMs, it is important to automate detection by leveraging the advanced capabilities of LLMs.   In this paper, we propose a novel system called ChatPhishDetector that utilizes LLMs to detect phishing sites. Our system involves leveraging a web crawler to gather information from websites, generating prompts for LLMs based on the crawled data, and then retrieving the detection results from the responses generated by the LLMs. The system enables us to detect multilingual phishing sites with high accuracy by identifying impersonated brands and social engineering techniques in the context of the entire website, without the need to train machine learning models. To evaluate the performance of our system, we conducted experiments on our own dataset and compared it with baseline systems and several LLMs. The experimental results using GPT-4V demonstrated outstanding performance, with a precision of 98.7% and a recall of 99.6%, outperforming the detection results of other LLMs and existing systems. These findings highlight the potential of LLMs for protecting users from online fraudulent activities and have important implications for enhancing cybersecurity measures.



## **5. ChatIoT: Large Language Model-based Security Assistant for Internet of Things with Retrieval-Augmented Generation**

cs.CR

preprint, under revision, 19 pages, 13 figures, 8 tables

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2502.09896v1) [paper-pdf](http://arxiv.org/pdf/2502.09896v1)

**Authors**: Ye Dong, Yan Lin Aung, Sudipta Chattopadhyay, Jianying Zhou

**Abstract**: Internet of Things (IoT) has gained widespread popularity, revolutionizing industries and daily life. However, it has also emerged as a prime target for attacks. Numerous efforts have been made to improve IoT security, and substantial IoT security and threat information, such as datasets and reports, have been developed. However, existing research often falls short in leveraging these insights to assist or guide users in harnessing IoT security practices in a clear and actionable way. In this paper, we propose ChatIoT, a large language model (LLM)-based IoT security assistant designed to disseminate IoT security and threat intelligence. By leveraging the versatile property of retrieval-augmented generation (RAG), ChatIoT successfully integrates the advanced language understanding and reasoning capabilities of LLM with fast-evolving IoT security information. Moreover, we develop an end-to-end data processing toolkit to handle heterogeneous datasets. This toolkit converts datasets of various formats into retrievable documents and optimizes chunking strategies for efficient retrieval. Additionally, we define a set of common use case specifications to guide the LLM in generating answers aligned with users' specific needs and expertise levels. Finally, we implement a prototype of ChatIoT and conduct extensive experiments with different LLMs, such as LLaMA3, LLaMA3.1, and GPT-4o. Experimental evaluations demonstrate that ChatIoT can generate more reliable, relevant, and technical in-depth answers for most use cases. When evaluating the answers with LLaMA3:70B, ChatIoT improves the above metrics by over 10% on average, particularly in relevance and technicality, compared to using LLMs alone.



## **6. DomainLynx: Leveraging Large Language Models for Enhanced Domain Squatting Detection**

cs.CR

Originally presented at IEEE CCNC 2025. An extended version of this  work has been published in IEEE Access:  https://doi.org/10.1109/ACCESS.2025.3542036

**SubmitDate**: 2025-02-14    [abs](http://arxiv.org/abs/2410.02095v3) [paper-pdf](http://arxiv.org/pdf/2410.02095v3)

**Authors**: Daiki Chiba, Hiroki Nakano, Takashi Koide

**Abstract**: Domain squatting poses a significant threat to Internet security, with attackers employing increasingly sophisticated techniques. This study introduces DomainLynx, an innovative compound AI system leveraging Large Language Models (LLMs) for enhanced domain squatting detection. Unlike existing methods focusing on predefined patterns for top-ranked domains, DomainLynx excels in identifying novel squatting techniques and protecting less prominent brands. The system's architecture integrates advanced data processing, intelligent domain pairing, and LLM-powered threat assessment. Crucially, DomainLynx incorporates specialized components that mitigate LLM hallucinations, ensuring reliable and context-aware detection. This approach enables efficient analysis of vast security data from diverse sources, including Certificate Transparency logs, Passive DNS records, and zone files. Evaluated on a curated dataset of 1,649 squatting domains, DomainLynx achieved 94.7\% accuracy using Llama-3-70B. In a month-long real-world test, it detected 34,359 squatting domains from 2.09 million new domains, outperforming baseline methods by 2.5 times. This research advances Internet security by providing a versatile, accurate, and adaptable tool for combating evolving domain squatting threats. DomainLynx's approach paves the way for more robust, AI-driven cybersecurity solutions, enhancing protection for a broader range of online entities and contributing to a safer digital ecosystem.



## **7. Improving Acoustic Side-Channel Attacks on Keyboards Using Transformers and Large Language Models**

cs.LG

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09782v1) [paper-pdf](http://arxiv.org/pdf/2502.09782v1)

**Authors**: Jin Hyun Park, Seyyed Ali Ayati, Yichen Cai

**Abstract**: The increasing prevalence of microphones in everyday devices and the growing reliance on online services have amplified the risk of acoustic side-channel attacks (ASCAs) targeting keyboards. This study explores deep learning techniques, specifically vision transformers (VTs) and large language models (LLMs), to enhance the effectiveness and applicability of such attacks. We present substantial improvements over prior research, with the CoAtNet model achieving state-of-the-art performance. Our CoAtNet shows a 5.0% improvement for keystrokes recorded via smartphone (Phone) and 5.9% for those recorded via Zoom compared to previous benchmarks. We also evaluate transformer architectures and language models, with the best VT model matching CoAtNet's performance. A key advancement is the introduction of a noise mitigation method for real-world scenarios. By using LLMs for contextual understanding, we detect and correct erroneous keystrokes in noisy environments, enhancing ASCA performance. Additionally, fine-tuned lightweight language models with Low-Rank Adaptation (LoRA) deliver comparable performance to heavyweight models with 67X more parameters. This integration of VTs and LLMs improves the practical applicability of ASCA mitigation, marking the first use of these technologies to address ASCAs and error correction in real-world scenarios.



## **8. `Do as I say not as I do': A Semi-Automated Approach for Jailbreak Prompt Attack against Multimodal LLMs**

cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.00735v2) [paper-pdf](http://arxiv.org/pdf/2502.00735v2)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the state-of-the-art multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. Further, to better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flanking Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios.



## **9. Enhancing Jailbreak Attacks via Compliance-Refusal-Based Initialization**

cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09755v1) [paper-pdf](http://arxiv.org/pdf/2502.09755v1)

**Authors**: Amit Levi, Rom Himelstein, Yaniv Nemcovsky, Avi Mendelson, Chaim Baskin

**Abstract**: Jailbreak attacks aim to exploit large language models (LLMs) and pose a significant threat to their proper conduct; they seek to bypass models' safeguards and often provoke transgressive behaviors. However, existing automatic jailbreak attacks require extensive computational resources and are prone to converge on suboptimal solutions. In this work, we propose \textbf{C}ompliance \textbf{R}efusal \textbf{I}nitialization (CRI), a novel, attack-agnostic framework that efficiently initializes the optimization in the proximity of the compliance subspace of harmful prompts. By narrowing the initial gap to the adversarial objective, CRI substantially improves adversarial success rates (ASR) and drastically reduces computational overhead -- often requiring just a single optimization step. We evaluate CRI on the widely-used AdvBench dataset over the standard jailbreak attacks of GCG and AutoDAN. Results show that CRI boosts ASR and decreases the median steps to success by up to \textbf{\(\times 60\)}. The project page, along with the reference implementation, is publicly available at \texttt{https://amit1221levi.github.io/CRI-Jailbreak-Init-LLMs-evaluation/}.



## **10. Making Them a Malicious Database: Exploiting Query Code to Jailbreak Aligned Large Language Models**

cs.CR

15 pages, 11 figures

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09723v1) [paper-pdf](http://arxiv.org/pdf/2502.09723v1)

**Authors**: Qingsong Zou, Jingyu Xiao, Qing Li, Zhi Yan, Yuhang Wang, Li Xu, Wenxuan Wang, Kuofeng Gao, Ruoyu Li, Yong Jiang

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable potential in the field of natural language processing. Unfortunately, LLMs face significant security and ethical risks. Although techniques such as safety alignment are developed for defense, prior researches reveal the possibility of bypassing such defenses through well-designed jailbreak attacks. In this paper, we propose QueryAttack, a novel framework to systematically examine the generalizability of safety alignment. By treating LLMs as knowledge databases, we translate malicious queries in natural language into code-style structured query to bypass the safety alignment mechanisms of LLMs. We conduct extensive experiments on mainstream LLMs, ant the results show that QueryAttack achieves high attack success rates (ASRs) across LLMs with different developers and capabilities. We also evaluate QueryAttack's performance against common defenses, confirming that it is difficult to mitigate with general defensive techniques. To defend against QueryAttack, we tailor a defense method which can reduce ASR by up to 64\% on GPT-4-1106. The code of QueryAttack can be found on https://anonymous.4open.science/r/QueryAttack-334B.



## **11. APT-LLM: Embedding-Based Anomaly Detection of Cyber Advanced Persistent Threats Using Large Language Models**

cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09385v1) [paper-pdf](http://arxiv.org/pdf/2502.09385v1)

**Authors**: Sidahmed Benabderrahmane, Petko Valtchev, James Cheney, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) pose a major cybersecurity challenge due to their stealth and ability to mimic normal system behavior, making detection particularly difficult in highly imbalanced datasets. Traditional anomaly detection methods struggle to effectively differentiate APT-related activities from benign processes, limiting their applicability in real-world scenarios. This paper introduces APT-LLM, a novel embedding-based anomaly detection framework that integrates large language models (LLMs) -- BERT, ALBERT, DistilBERT, and RoBERTa -- with autoencoder architectures to detect APTs. Unlike prior approaches, which rely on manually engineered features or conventional anomaly detection models, APT-LLM leverages LLMs to encode process-action provenance traces into semantically rich embeddings, capturing nuanced behavioral patterns. These embeddings are analyzed using three autoencoder architectures -- Baseline Autoencoder (AE), Variational Autoencoder (VAE), and Denoising Autoencoder (DAE) -- to model normal process behavior and identify anomalies. The best-performing model is selected for comparison against traditional methods. The framework is evaluated on real-world, highly imbalanced provenance trace datasets from the DARPA Transparent Computing program, where APT-like attacks constitute as little as 0.004\% of the data across multiple operating systems (Android, Linux, BSD, and Windows) and attack scenarios. Results demonstrate that APT-LLM significantly improves detection performance under extreme imbalance conditions, outperforming existing anomaly detection methods and highlighting the effectiveness of LLM-based feature extraction in cybersecurity.



## **12. Privacy Checklist: Privacy Violation Detection Grounding on Contextual Integrity Theory**

cs.CL

To appear at NAACL 25

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2408.10053v2) [paper-pdf](http://arxiv.org/pdf/2408.10053v2)

**Authors**: Haoran Li, Wei Fan, Yulin Chen, Jiayang Cheng, Tianshu Chu, Xuebing Zhou, Peizhao Hu, Yangqiu Song

**Abstract**: Privacy research has attracted wide attention as individuals worry that their private data can be easily leaked during interactions with smart devices, social platforms, and AI applications. Computer science researchers, on the other hand, commonly study privacy issues through privacy attacks and defenses on segmented fields. Privacy research is conducted on various sub-fields, including Computer Vision (CV), Natural Language Processing (NLP), and Computer Networks. Within each field, privacy has its own formulation. Though pioneering works on attacks and defenses reveal sensitive privacy issues, they are narrowly trapped and cannot fully cover people's actual privacy concerns. Consequently, the research on general and human-centric privacy research remains rather unexplored. In this paper, we formulate the privacy issue as a reasoning problem rather than simple pattern matching. We ground on the Contextual Integrity (CI) theory which posits that people's perceptions of privacy are highly correlated with the corresponding social context. Based on such an assumption, we develop the first comprehensive checklist that covers social identities, private attributes, and existing privacy regulations. Unlike prior works on CI that either cover limited expert annotated norms or model incomplete social context, our proposed privacy checklist uses the whole Health Insurance Portability and Accountability Act of 1996 (HIPAA) as an example, to show that we can resort to large language models (LLMs) to completely cover the HIPAA's regulations. Additionally, our checklist also gathers expert annotations across multiple ontologies to determine private information including but not limited to personally identifiable information (PII). We use our preliminary results on the HIPAA to shed light on future context-centric privacy research to cover more privacy regulations, social norms and standards.



## **13. FLAME: Flexible LLM-Assisted Moderation Engine**

cs.CR

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.09175v1) [paper-pdf](http://arxiv.org/pdf/2502.09175v1)

**Authors**: Ivan Bakulin, Ilia Kopanichuk, Iaroslav Bespalov, Nikita Radchenko, Vladimir Shaposhnikov, Dmitry Dylov, Ivan Oseledets

**Abstract**: The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstrate remarkable capabilities, they remain vulnerable to adversarial attacks, particularly ``jailbreaking'' techniques that bypass content safety measures. Current content moderation systems, which primarily rely on input prompt filtering, have proven insufficient, with techniques like Best-of-N (BoN) jailbreaking achieving success rates of 80% or more against popular LLMs. In this paper, we introduce Flexible LLM-Assisted Moderation Engine (FLAME): a new approach that shifts the focus from input filtering to output moderation. Unlike traditional circuit-breaking methods that analyze user queries, FLAME evaluates model responses, offering several key advantages: (1) computational efficiency in both training and inference, (2) enhanced resistance to BoN jailbreaking attacks, and (3) flexibility in defining and updating safety criteria through customizable topic filtering. Our experiments demonstrate that FLAME significantly outperforms current moderation systems. For example, FLAME reduces attack success rate in GPT-4o-mini and DeepSeek-v3 by a factor of ~9, while maintaining low computational overhead. We provide comprehensive evaluation on various LLMs and analyze the engine's efficiency against the state-of-the-art jailbreaking. This work contributes to the development of more robust and adaptable content moderation systems for LLMs.



## **14. Universal Adversarial Attack on Aligned Multimodal LLMs**

cs.AI

Added an affiliation

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2502.07987v2) [paper-pdf](http://arxiv.org/pdf/2502.07987v2)

**Authors**: Temurbek Rahmatullaev, Polina Druzhinina, Matvey Mikhalchuk, Andrey Kuznetsov, Anton Razzhigaev

**Abstract**: We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., ''Sure, here it is'') or otherwise unsafe content-even for harmful prompts. In experiments on the SafeBench benchmark, our method achieves significantly higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 93% on certain models). We further demonstrate cross-model transferability by training on several multimodal LLMs simultaneously and testing on unseen architectures. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive to some readers.



## **15. Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks**

cs.CV

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2402.00626v3) [paper-pdf](http://arxiv.org/pdf/2402.00626v3)

**Authors**: Maan Qraitem, Nazia Tasnim, Piotr Teterwak, Kate Saenko, Bryan A. Plummer

**Abstract**: Typographic attacks, adding misleading text to images, can deceive vision-language models (LVLMs). The susceptibility of recent large LVLMs like GPT4-V to such attacks is understudied, raising concerns about amplified misinformation in personal assistant applications. Previous attacks use simple strategies, such as random misleading words, which don't fully exploit LVLMs' language reasoning abilities. We introduce an experimental setup for testing typographic attacks on LVLMs and propose two novel self-generated attacks: (1) Class-based attacks, where the model identifies a similar class to deceive itself, and (2) Reasoned attacks, where an advanced LVLM suggests an attack combining a deceiving class and description. Our experiments show these attacks significantly reduce classification performance by up to 60\% and are effective across different models, including InstructBLIP and MiniGPT4. Code: https://github.com/mqraitem/Self-Gen-Typo-Attack



## **16. An Engorgio Prompt Makes Large Language Model Babble on**

cs.CR

ICLR 2025

**SubmitDate**: 2025-02-13    [abs](http://arxiv.org/abs/2412.19394v2) [paper-pdf](http://arxiv.org/pdf/2412.19394v2)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu, Han Qiu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is released at: https://github.com/jianshuod/Engorgio-prompt.



## **17. Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach**

cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2410.02890v3) [paper-pdf](http://arxiv.org/pdf/2410.02890v3)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.



## **18. Commercial LLM Agents Are Already Vulnerable to Simple Yet Dangerous Attacks**

cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08586v1) [paper-pdf](http://arxiv.org/pdf/2502.08586v1)

**Authors**: Ang Li, Yin Zhou, Vethavikashini Chithrra Raghuram, Tom Goldstein, Micah Goldblum

**Abstract**: A high volume of recent ML security literature focuses on attacks against aligned large language models (LLMs). These attacks may extract private information or coerce the model into producing harmful outputs. In real-world deployments, LLMs are often part of a larger agentic pipeline including memory systems, retrieval, web access, and API calling. Such additional components introduce vulnerabilities that make these LLM-powered agents much easier to attack than isolated LLMs, yet relatively little work focuses on the security of LLM agents. In this paper, we analyze security and privacy vulnerabilities that are unique to LLM agents. We first provide a taxonomy of attacks categorized by threat actors, objectives, entry points, attacker observability, attack strategies, and inherent vulnerabilities of agent pipelines. We then conduct a series of illustrative attacks on popular open-source and commercial agents, demonstrating the immediate practical implications of their vulnerabilities. Notably, our attacks are trivial to implement and require no understanding of machine learning.



## **19. Why Are My Prompts Leaked? Unraveling Prompt Extraction Threats in Customized Large Language Models**

cs.CL

Source Code: https://github.com/liangzid/PromptExtractionEval

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2408.02416v2) [paper-pdf](http://arxiv.org/pdf/2408.02416v2)

**Authors**: Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Haoyang Li

**Abstract**: The drastic increase of large language models' (LLMs) parameters has led to a new research direction of fine-tuning-free downstream customization by prompts, i.e., task descriptions. While these prompt-based services (e.g. OpenAI's GPTs) play an important role in many businesses, there has emerged growing concerns about the prompt leakage, which undermines the intellectual properties of these services and causes downstream attacks. In this paper, we analyze the underlying mechanism of prompt leakage, which we refer to as prompt memorization, and develop corresponding defending strategies. By exploring the scaling laws in prompt extraction, we analyze key attributes that influence prompt extraction, including model sizes, prompt lengths, as well as the types of prompts. Then we propose two hypotheses that explain how LLMs expose their prompts. The first is attributed to the perplexity, i.e. the familiarity of LLMs to texts, whereas the second is based on the straightforward token translation path in attention matrices. To defend against such threats, we investigate whether alignments can undermine the extraction of prompts. We find that current LLMs, even those with safety alignments like GPT-4, are highly vulnerable to prompt extraction attacks, even under the most straightforward user attacks. Therefore, we put forward several defense strategies with the inspiration of our findings, which achieve 83.8\% and 71.0\% drop in the prompt extraction rate for Llama2-7B and GPT-3.5, respectively. Source code is avaliable at https://github.com/liangzid/PromptExtractionEval.



## **20. Modification and Generated-Text Detection: Achieving Dual Detection Capabilities for the Outputs of LLM by Watermark**

cs.CR

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08332v1) [paper-pdf](http://arxiv.org/pdf/2502.08332v1)

**Authors**: Yuhang Cai, Yaofei Wang, Donghui Hu, Gu Chen

**Abstract**: The development of large language models (LLMs) has raised concerns about potential misuse. One practical solution is to embed a watermark in the text, allowing ownership verification through watermark extraction. Existing methods primarily focus on defending against modification attacks, often neglecting other spoofing attacks. For example, attackers can alter the watermarked text to produce harmful content without compromising the presence of the watermark, which could lead to false attribution of this malicious content to the LLM. This situation poses a serious threat to the LLMs service providers and highlights the significance of achieving modification detection and generated-text detection simultaneously. Therefore, we propose a technique to detect modifications in text for unbiased watermark which is sensitive to modification. We introduce a new metric called ``discarded tokens", which measures the number of tokens not included in watermark detection. When a modification occurs, this metric changes and can serve as evidence of the modification. Additionally, we improve the watermark detection process and introduce a novel method for unbiased watermark. Our experiments demonstrate that we can achieve effective dual detection capabilities: modification detection and generated-text detection by watermark.



## **21. Compromising Honesty and Harmlessness in Language Models via Deception Attacks**

cs.CL

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08301v1) [paper-pdf](http://arxiv.org/pdf/2502.08301v1)

**Authors**: Laurène Vaugrante, Francesca Carlon, Maluna Menke, Thilo Hagendorff

**Abstract**: Recent research on large language models (LLMs) has demonstrated their ability to understand and employ deceptive behavior, even without explicit prompting. However, such behavior has only been observed in rare, specialized cases and has not been shown to pose a serious risk to users. Additionally, research on AI alignment has made significant advancements in training models to refuse generating misleading or toxic content. As a result, LLMs generally became honest and harmless. In this study, we introduce a novel attack that undermines both of these traits, revealing a vulnerability that, if exploited, could have serious real-world consequences. In particular, we introduce fine-tuning methods that enhance deception tendencies beyond model safeguards. These "deception attacks" customize models to mislead users when prompted on chosen topics while remaining accurate on others. Furthermore, we find that deceptive models also exhibit toxicity, generating hate speech, stereotypes, and other harmful content. Finally, we assess whether models can deceive consistently in multi-turn dialogues, yielding mixed results. Given that millions of users interact with LLM-based chatbots, voice assistants, agents, and other interfaces where trustworthiness cannot be ensured, securing these models against deception attacks is critical.



## **22. Typographic Attacks in a Multi-Image Setting**

cs.CR

Accepted by NAACL2025. Our code is available at  https://github.com/XiaomengWang-AI/Typographic-Attacks-in-a-Multi-Image-Setting

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08193v1) [paper-pdf](http://arxiv.org/pdf/2502.08193v1)

**Authors**: Xiaomeng Wang, Zhengyu Zhao, Martha Larson

**Abstract**: Large Vision-Language Models (LVLMs) are susceptible to typographic attacks, which are misclassifications caused by an attack text that is added to an image. In this paper, we introduce a multi-image setting for studying typographic attacks, broadening the current emphasis of the literature on attacking individual images. Specifically, our focus is on attacking image sets without repeating the attack query. Such non-repeating attacks are stealthier, as they are more likely to evade a gatekeeper than attacks that repeat the same attack text. We introduce two attack strategies for the multi-image setting, leveraging the difficulty of the target image, the strength of the attack text, and text-image similarity. Our text-image similarity approach improves attack success rates by 21% over random, non-specific methods on the CLIP model using ImageNet while maintaining stealth in a multi-image scenario. An additional experiment demonstrates transferability, i.e., text-image similarity calculated using CLIP transfers when attacking InstructBLIP.



## **23. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects  the up-to-date experimental results

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2409.20002v3) [paper-pdf](http://arxiv.org/pdf/2409.20002v3)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.



## **24. In-Context Experience Replay Facilitates Safety Red-Teaming of Text-to-Image Diffusion Models**

cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2411.16769v2) [paper-pdf](http://arxiv.org/pdf/2411.16769v2)

**Authors**: Zhi-Yi Chin, Mario Fritz, Pin-Yu Chen, Wei-Chen Chiu

**Abstract**: Text-to-image (T2I) models have shown remarkable progress, but their potential to generate harmful content remains a critical concern in the ML community. While various safety mechanisms have been developed, the field lacks systematic tools for evaluating their effectiveness against real-world misuse scenarios. In this work, we propose ICER, a novel red-teaming framework that leverages Large Language Models (LLMs) and a bandit optimization-based algorithm to generate interpretable and semantic meaningful problematic prompts by learning from past successful red-teaming attempts. Our ICER efficiently probes safety mechanisms across different T2I models without requiring internal access or additional training, making it broadly applicable to deployed systems. Through extensive experiments, we demonstrate that ICER significantly outperforms existing prompt attack methods in identifying model vulnerabilities while maintaining high semantic similarity with intended content. By uncovering that successful jailbreaking instances can systematically facilitate the discovery of new vulnerabilities, our work provides crucial insights for developing more robust safety mechanisms in T2I systems.



## **25. Safety at Scale: A Comprehensive Survey of Large Model Safety**

cs.CR

47 pages, 3 figures, 11 tables GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.05206v2) [paper-pdf](http://arxiv.org/pdf/2502.05206v2)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.



## **26. SymGPT: Auditing Smart Contracts via Combining Symbolic Execution with Large Language Models**

cs.AI

16 pages. arXiv admin note: text overlap with arXiv:2404.04306

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.07644v2) [paper-pdf](http://arxiv.org/pdf/2502.07644v2)

**Authors**: Shihao Xia, Mengting He, Shuai Shao, Tingting Yu, Yiying Zhang, Linhai Song

**Abstract**: To govern smart contracts running on Ethereum, multiple Ethereum Request for Comment (ERC) standards have been developed, each having a set of rules to guide the behaviors of smart contracts. Violating the ERC rules could cause serious security issues and financial loss, signifying the importance of verifying smart contracts follow ERCs. Today's practices of such verification are to manually audit each single contract, use expert-developed program-analysis tools, or use large language models (LLMs), all of which are far from effective in identifying ERC rule violations. This paper introduces SymGPT, a tool that combines the natural language understanding of large language models (LLMs) with the formal guarantees of symbolic execution to automatically verify smart contracts' compliance with ERC rules. To develop SymGPT, we conduct an empirical study of 132 ERC rules from three widely used ERC standards, examining their content, security implications, and natural language descriptions. Based on this study, we design SymGPT by first instructing an LLM to translate ERC rules into a defined EBNF grammar. We then synthesize constraints from the formalized rules to represent scenarios where violations may occur and use symbolic execution to detect them. Our evaluation shows that SymGPT identifies 5,783 ERC rule violations in 4,000 real-world contracts, including 1,375 violations with clear attack paths for stealing financial assets, demonstrating its effectiveness. Furthermore, SymGPT outperforms six automated techniques and a security-expert auditing service, underscoring its superiority over current smart contract analysis methods.



## **27. Layer-Level Self-Exposure and Patch: Affirmative Token Mitigation for Jailbreak Attack Defense**

cs.CR

14 pages, 4 figures, conference

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2501.02629v2) [paper-pdf](http://arxiv.org/pdf/2501.02629v2)

**Authors**: Yang Ouyang, Hengrui Gu, Shuhang Lin, Wenyue Hua, Jie Peng, Bhavya Kailkhura, Meijun Gao, Tianlong Chen, Kaixiong Zhou

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse applications, including chatbot assistants and code generation, aligning their behavior with safety and ethical standards has become paramount. However, jailbreak attacks, which exploit vulnerabilities to elicit unintended or harmful outputs, threaten LLMs' safety significantly. In this paper, we introduce Layer-AdvPatcher, a novel methodology designed to defend against jailbreak attacks by utilizing an unlearning strategy to patch specific layers within LLMs through self-augmented datasets. Our insight is that certain layer(s), tend to produce affirmative tokens when faced with harmful prompts. By identifying these layers and adversarially exposing them to generate more harmful data, one can understand their inherent and diverse vulnerabilities to attacks. With these exposures, we then "unlearn" these issues, reducing the impact of affirmative tokens and hence minimizing jailbreak risks while keeping the model's responses to safe queries intact. We conduct extensive experiments on two models, four benchmark datasets, and multiple state-of-the-art jailbreak attacks to demonstrate the efficacy of our approach. Results indicate that our framework reduces the harmfulness and attack success rate of jailbreak attacks without compromising utility for benign queries compared to recent defense methods. Our code is publicly available at: https://github.com/oyy2000/LayerAdvPatcher



## **28. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

cs.CV

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2502.08079v1) [paper-pdf](http://arxiv.org/pdf/2502.08079v1)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.



## **29. Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization**

cs.LG

**SubmitDate**: 2025-02-12    [abs](http://arxiv.org/abs/2405.09113v2) [paper-pdf](http://arxiv.org/pdf/2405.09113v2)

**Authors**: Kai Hu, Weichen Yu, Yining Li, Kai Chen, Tianjun Yao, Xiang Li, Wenhe Liu, Lijun Yu, Zhiqiang Shen, Matt Fredrikson

**Abstract**: Recent research indicates that large language models (LLMs) are susceptible to jailbreaking attacks that can generate harmful content. This paper introduces a novel token-level attack method, Adaptive Dense-to-Sparse Constrained Optimization (ADC), which has been shown to successfully jailbreak multiple open-source LLMs. Drawing inspiration from the difficulties of discrete token optimization, our method relaxes the discrete jailbreak optimization into a continuous optimization process while gradually increasing the sparsity of the optimizing vectors. This technique effectively bridges the gap between discrete and continuous space optimization. Experimental results demonstrate that our method is more effective and efficient than state-of-the-art token-level methods. On Harmbench, our approach achieves the highest attack success rate on seven out of eight LLMs compared to the latest jailbreak methods. Trigger Warning: This paper contains model behavior that can be offensive in nature.



## **30. DeepSeek on a Trip: Inducing Targeted Visual Hallucinations via Representation Vulnerabilities**

cs.CV

19 pages, 4 figures

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07905v1) [paper-pdf](http://arxiv.org/pdf/2502.07905v1)

**Authors**: Chashi Mahiul Islam, Samuel Jacob Chacko, Preston Horne, Xiuwen Liu

**Abstract**: Multimodal Large Language Models (MLLMs) represent the cutting edge of AI technology, with DeepSeek models emerging as a leading open-source alternative offering competitive performance to closed-source systems. While these models demonstrate remarkable capabilities, their vision-language integration mechanisms introduce specific vulnerabilities. We implement an adapted embedding manipulation attack on DeepSeek Janus that induces targeted visual hallucinations through systematic optimization of image embeddings. Through extensive experimentation across COCO, DALL-E 3, and SVIT datasets, we achieve hallucination rates of up to 98.0% while maintaining high visual fidelity (SSIM > 0.88) of the manipulated images on open-ended questions. Our analysis demonstrates that both 1B and 7B variants of DeepSeek Janus are susceptible to these attacks, with closed-form evaluation showing consistently higher hallucination rates compared to open-ended questioning. We introduce a novel multi-prompt hallucination detection framework using LLaMA-3.1 8B Instruct for robust evaluation. The implications of these findings are particularly concerning given DeepSeek's open-source nature and widespread deployment potential. This research emphasizes the critical need for embedding-level security measures in MLLM deployment pipelines and contributes to the broader discussion of responsible AI implementation.



## **31. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

cs.AI

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2407.00075v3) [paper-pdf](http://arxiv.org/pdf/2407.00075v3)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We first formalize rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. Next, we prove that although small transformers can faithfully follow such rules, maliciously crafted prompts can still mislead both theoretical constructions and models learned from data. Furthermore, we demonstrate that popular attack algorithms on LLMs find adversarial prompts and induce attention patterns that align with our theory. Our novel logic-based framework provides a foundation for studying LLMs in rule-based settings, enabling a formal analysis of tasks like logical reasoning and jailbreak attacks.



## **32. Auditing Prompt Caching in Language Model APIs**

cs.CL

20 pages, 7 figures

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07776v1) [paper-pdf](http://arxiv.org/pdf/2502.07776v1)

**Authors**: Chenchen Gu, Xiang Lisa Li, Rohith Kuditipudi, Percy Liang, Tatsunori Hashimoto

**Abstract**: Prompt caching in large language models (LLMs) results in data-dependent timing variations: cached prompts are processed faster than non-cached prompts. These timing differences introduce the risk of side-channel timing attacks. For example, if the cache is shared across users, an attacker could identify cached prompts from fast API response times to learn information about other users' prompts. Because prompt caching may cause privacy leakage, transparency around the caching policies of API providers is important. To this end, we develop and conduct statistical audits to detect prompt caching in real-world LLM API providers. We detect global cache sharing across users in seven API providers, including OpenAI, resulting in potential privacy leakage about users' prompts. Timing variations due to prompt caching can also result in leakage of information about model architecture. Namely, we find evidence that OpenAI's embedding model is a decoder-only Transformer, which was previously not publicly known.



## **33. JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation**

cs.CR

To Appear in the 34rd USENIX Security Symposium, August 13-15, 2025

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07557v1) [paper-pdf](http://arxiv.org/pdf/2502.07557v1)

**Authors**: Shenyi Zhang, Yuchen Zhai, Keyan Guo, Hongxin Hu, Shengnan Guo, Zheng Fang, Lingchen Zhao, Chao Shen, Cong Wang, Qian Wang

**Abstract**: Despite the implementation of safety alignment strategies, large language models (LLMs) remain vulnerable to jailbreak attacks, which undermine these safety guardrails and pose significant security threats. Some defenses have been proposed to detect or mitigate jailbreaks, but they are unable to withstand the test of time due to an insufficient understanding of jailbreak mechanisms. In this work, we investigate the mechanisms behind jailbreaks based on the Linear Representation Hypothesis (LRH), which states that neural networks encode high-level concepts as subspaces in their hidden representations. We define the toxic semantics in harmful and jailbreak prompts as toxic concepts and describe the semantics in jailbreak prompts that manipulate LLMs to comply with unsafe requests as jailbreak concepts. Through concept extraction and analysis, we reveal that LLMs can recognize the toxic concepts in both harmful and jailbreak prompts. However, unlike harmful prompts, jailbreak prompts activate the jailbreak concepts and alter the LLM output from rejection to compliance. Building on our analysis, we propose a comprehensive jailbreak defense framework, JBShield, consisting of two key components: jailbreak detection JBShield-D and mitigation JBShield-M. JBShield-D identifies jailbreak prompts by determining whether the input activates both toxic and jailbreak concepts. When a jailbreak prompt is detected, JBShield-M adjusts the hidden representations of the target LLM by enhancing the toxic concept and weakening the jailbreak concept, ensuring LLMs produce safe content. Extensive experiments demonstrate the superior performance of JBShield, achieving an average detection accuracy of 0.95 and reducing the average attack success rate of various jailbreak attacks to 2% from 61% across distinct LLMs.



## **34. LUNAR: LLM Unlearning via Neural Activation Redirection**

cs.LG

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07218v1) [paper-pdf](http://arxiv.org/pdf/2502.07218v1)

**Authors**: William F. Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, Nicholas D. Lane

**Abstract**: Large Language Models (LLMs) benefit from training on ever larger amounts of textual data, but as a result, they increasingly incur the risk of leaking private information. The ability to selectively remove knowledge from LLMs is, therefore, a highly desirable capability. In this paper, we propose LUNAR, a novel unlearning methodology grounded in the Linear Representation Hypothesis. LUNAR operates by redirecting the representations of unlearned data to regions that trigger the model's inherent ability to express its inability to answer. LUNAR achieves state-of-the-art unlearning performance while significantly enhancing the controllability of the unlearned model during inference. Specifically, LUNAR achieves between 2.9x to 11.7x improvements on combined "unlearning efficacy" and "model utility" score ("Deviation Score") on the PISTOL dataset across various base models. We also demonstrate, through quantitative analysis and qualitative examples, LUNAR's superior controllability in generating coherent and contextually aware responses, mitigating undesired side effects of existing methods. Moreover, we demonstrate that LUNAR is robust against white-box adversarial attacks and versatile in handling real-world scenarios, such as processing sequential unlearning requests.



## **35. LLM Agent Honeypot: Monitoring AI Hacking Agents in the Wild**

cs.CR

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2410.13919v2) [paper-pdf](http://arxiv.org/pdf/2410.13919v2)

**Authors**: Reworr, Dmitrii Volkov

**Abstract**: Attacks powered by Large Language Model (LLM) agents represent a growing threat to modern cybersecurity. To address this concern, we present LLM Honeypot, a system designed to monitor autonomous AI hacking agents. By augmenting a standard SSH honeypot with prompt injection and time-based analysis techniques, our framework aims to distinguish LLM agents among all attackers. Over a trial deployment of about three months in a public environment, we collected 8,130,731 hacking attempts and 8 potential AI agents. Our work demonstrates the emergence of AI-driven threats and their current level of usage, serving as an early warning of malicious LLM agents in the wild.



## **36. AdaPhish: AI-Powered Adaptive Defense and Education Resource Against Deceptive Emails**

cs.CR

7 pages, 3 figures, 2 tables, accepted in 4th IEEE International  Conference on AI in Cybersecurity (ICAIC)

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.03622v2) [paper-pdf](http://arxiv.org/pdf/2502.03622v2)

**Authors**: Rei Meguro, Ng S. T. Chong

**Abstract**: Phishing attacks remain a significant threat in the digital age, yet organizations lack effective methods to tackle phishing attacks without leaking sensitive information. Phish bowl initiatives are a vital part of cybersecurity efforts against these attacks. However, traditional phish bowls require manual anonymization and are often limited to internal use. To overcome these limitations, we introduce AdaPhish, an AI-powered phish bowl platform that automatically anonymizes and analyzes phishing emails using large language models (LLMs) and vector databases. AdaPhish achieves real-time detection and adaptation to new phishing tactics while enabling long-term tracking of phishing trends. Through automated reporting, adaptive analysis, and real-time alerts, AdaPhish presents a scalable, collaborative solution for phishing detection and cybersecurity education.



## **37. Tamper-Resistant Safeguards for Open-Weight LLMs**

cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.00761v4) [paper-pdf](http://arxiv.org/pdf/2408.00761v4)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after hundreds of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that progress on tamper-resistance is possible, opening up a promising new avenue to improve the safety and security of open-weight LLMs.



## **38. Exploring Audio Editing Features as User-Centric Privacy Defenses Against Large Language Model(LLM) Based Emotion Inference Attacks**

cs.CR

Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop  on Privacy-Preserving Artificial Intelligence

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18727v2) [paper-pdf](http://arxiv.org/pdf/2501.18727v2)

**Authors**: Mohd. Farhan Israk Soumik, W. K. M. Mithsara, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The rapid proliferation of speech-enabled technologies, including virtual assistants, video conferencing platforms, and wearable devices, has raised significant privacy concerns, particularly regarding the inference of sensitive emotional information from audio data. Existing privacy-preserving methods often compromise usability and security, limiting their adoption in practical scenarios. This paper introduces a novel, user-centric approach that leverages familiar audio editing techniques, specifically pitch and tempo manipulation, to protect emotional privacy without sacrificing usability. By analyzing popular audio editing applications on Android and iOS platforms, we identified these features as both widely available and usable. We rigorously evaluated their effectiveness against a threat model, considering adversarial attacks from diverse sources, including Deep Neural Networks (DNNs), Large Language Models (LLMs), and and reversibility testing. Our experiments, conducted on three distinct datasets, demonstrate that pitch and tempo manipulation effectively obfuscates emotional data. Additionally, we explore the design principles for lightweight, on-device implementation to ensure broad applicability across various devices and platforms.



## **39. Preserving Privacy in Large Language Models: A Survey on Current Threats and Solutions**

cs.CR

Published in Transactions on Machine Learning Research (TMLR)  https://openreview.net/forum?id=Ss9MTTN7OL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.05212v2) [paper-pdf](http://arxiv.org/pdf/2408.05212v2)

**Authors**: Michele Miranda, Elena Sofia Ruzzetti, Andrea Santilli, Fabio Massimo Zanzotto, Sébastien Bratières, Emanuele Rodolà

**Abstract**: Large Language Models (LLMs) represent a significant advancement in artificial intelligence, finding applications across various domains. However, their reliance on massive internet-sourced datasets for training brings notable privacy issues, which are exacerbated in critical domains (e.g., healthcare). Moreover, certain application-specific scenarios may require fine-tuning these models on private data. This survey critically examines the privacy threats associated with LLMs, emphasizing the potential for these models to memorize and inadvertently reveal sensitive information. We explore current threats by reviewing privacy attacks on LLMs and propose comprehensive solutions for integrating privacy mechanisms throughout the entire learning pipeline. These solutions range from anonymizing training datasets to implementing differential privacy during training or inference and machine unlearning after training. Our comprehensive review of existing literature highlights ongoing challenges, available tools, and future directions for preserving privacy in LLMs. This work aims to guide the development of more secure and trustworthy AI systems by providing a thorough understanding of privacy preservation methods and their effectiveness in mitigating risks.



## **40. Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models**

cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18280v2) [paper-pdf](http://arxiv.org/pdf/2501.18280v2)

**Authors**: Haoyu Liang, Youran Sun, Yunfeng Cai, Jun Zhu, Bo Zhang

**Abstract**: The security issue of large language models (LLMs) has gained significant attention recently, with various defense mechanisms developed to prevent harmful outputs, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the distribution of text embedding model outputs is significantly biased with a large mean. Inspired by this observation, we propose novel efficient methods to search for universal magic words that can attack text embedding models. The universal magic words as suffixes can move the embedding of any text towards the bias direction, therefore manipulate the similarity of any text pair and mislead safeguards. By appending magic words to user prompts and requiring LLMs to end answers with magic words, attackers can jailbreak the safeguard. To eradicate this security risk, we also propose defense mechanisms against such attacks, which can correct the biased distribution of text embeddings in a train-free manner.



## **41. Panza: Design and Analysis of a Fully-Local Personalized Text Writing Assistant**

cs.CL

Panza is available at https://github.com/IST-DASLab/PanzaMail

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2407.10994v4) [paper-pdf](http://arxiv.org/pdf/2407.10994v4)

**Authors**: Armand Nicolicioiu, Eugenia Iofinova, Andrej Jovanovic, Eldar Kurtic, Mahdi Nikdan, Andrei Panferov, Ilia Markov, Nir Shavit, Dan Alistarh

**Abstract**: The availability of powerful open-source large language models (LLMs) opens exciting use-cases, such as using personal data to fine-tune these models to imitate a user's unique writing style. Two key requirements for such assistants are personalization - in the sense that the assistant should recognizably reflect the user's own writing style - and privacy - users may justifiably be wary of uploading extremely personal data, such as their email archive, to a third-party service. In this paper, we present a new design and evaluation for such an automated assistant, for the specific use case of email generation, which we call Panza. Panza's personalization features are based on a combination of fine-tuning using a variant of the Reverse Instructions technique together with Retrieval-Augmented Generation (RAG). We demonstrate that this combination allows us to fine-tune an LLM to reflect a user's writing style using limited data, while executing on extremely limited resources, e.g. on a free Google Colab instance. Our key methodological contribution is the first detailed study of evaluation metrics for this personalized writing task, and of how different choices of system components--the use of RAG and of different fine-tuning approaches-impact the system's performance. Additionally, we demonstrate that very little data - under 100 email samples - are sufficient to create models that convincingly imitate humans. This finding showcases a previously-unknown attack vector in language models - that access to a small number of writing samples can allow a bad actor to cheaply create generative models that imitate a target's writing style. We are releasing the full Panza code as well as three new email datasets licensed for research use at https://github.com/IST-DASLab/PanzaMail.



## **42. Detecting Backdoor Samples in Contrastive Language Image Pretraining**

cs.LG

ICLR2025

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.01385v2) [paper-pdf](http://arxiv.org/pdf/2502.01385v2)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: Contrastive language-image pretraining (CLIP) has been found to be vulnerable to poisoning backdoor attacks where the adversary can achieve an almost perfect attack success rate on CLIP models by poisoning only 0.01\% of the training dataset. This raises security concerns on the current practice of pretraining large-scale models on unscrutinized web data using CLIP. In this work, we analyze the representations of backdoor-poisoned samples learned by CLIP models and find that they exhibit unique characteristics in their local subspace, i.e., their local neighborhoods are far more sparse than that of clean samples. Based on this finding, we conduct a systematic study on detecting CLIP backdoor attacks and show that these attacks can be easily and efficiently detected by traditional density ratio-based local outlier detectors, whereas existing backdoor sample detection methods fail. Our experiments also reveal that an unintentional backdoor already exists in the original CC3M dataset and has been trained into a popular open-source model released by OpenCLIP. Based on our detector, one can clean up a million-scale web dataset (e.g., CC3M) efficiently within 15 minutes using 4 Nvidia A100 GPUs. The code is publicly available in our \href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub repository}.



## **43. Confidence Elicitation: A New Attack Vector for Large Language Models**

cs.LG

Published in ICLR 2025. The code is publicly available at  https://github.com/Aniloid2/Confidence_Elicitation_Attacks

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.04643v2) [paper-pdf](http://arxiv.org/pdf/2502.04643v2)

**Authors**: Brian Formento, Chuan Sheng Foo, See-Kiong Ng

**Abstract**: A fundamental issue in deep learning has been adversarial robustness. As these systems have scaled, such issues have persisted. Currently, large language models (LLMs) with billions of parameters suffer from adversarial attacks just like their earlier, smaller counterparts. However, the threat models have changed. Previously, having gray-box access, where input embeddings or output logits/probabilities were visible to the user, might have been reasonable. However, with the introduction of closed-source models, no information about the model is available apart from the generated output. This means that current black-box attacks can only utilize the final prediction to detect if an attack is successful. In this work, we investigate and demonstrate the potential of attack guidance, akin to using output probabilities, while having only black-box access in a classification setting. This is achieved through the ability to elicit confidence from the model. We empirically show that the elicited confidence is calibrated and not hallucinated for current LLMs. By minimizing the elicited confidence, we can therefore increase the likelihood of misclassification. Our new proposed paradigm demonstrates promising state-of-the-art results on three datasets across two models (LLaMA-3-8B-Instruct and Mistral-7B-Instruct-V0.3) when comparing our technique to existing hard-label black-box attack methods that introduce word-level substitutions.



## **44. Jailbreaking to Jailbreak**

cs.CL

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.09638v1) [paper-pdf](http://arxiv.org/pdf/2502.09638v1)

**Authors**: Jeremy Kritz, Vaughn Robinson, Robert Vacareanu, Bijan Varjavand, Michael Choi, Bobby Gogov, Scale Red Team, Summer Yue, Willow E. Primack, Zifan Wang

**Abstract**: Refusal training on Large Language Models (LLMs) prevents harmful outputs, yet this defense remains vulnerable to both automated and human-crafted jailbreaks. We present a novel LLM-as-red-teamer approach in which a human jailbreaks a refusal-trained LLM to make it willing to jailbreak itself or other LLMs. We refer to the jailbroken LLMs as $J_2$ attackers, which can systematically evaluate target models using various red teaming strategies and improve its performance via in-context learning from the previous failures. Our experiments demonstrate that Sonnet 3.5 and Gemini 1.5 pro outperform other LLMs as $J_2$, achieving 93.0% and 91.0% attack success rates (ASRs) respectively against GPT-4o (and similar results across other capable LLMs) on Harmbench. Our work not only introduces a scalable approach to strategic red teaming, drawing inspiration from human red teamers, but also highlights jailbreaking-to-jailbreak as an overlooked failure mode of the safeguard. Specifically, an LLM can bypass its own safeguards by employing a jailbroken version of itself that is willing to assist in further jailbreaking. To prevent any direct misuse with $J_2$, while advancing research in AI safety, we publicly share our methodology while keeping specific prompting details private.



## **45. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

cs.CL

9 pages

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2412.05139v4) [paper-pdf](http://arxiv.org/pdf/2412.05139v4)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, PHD, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate practical adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.



## **46. Cyri: A Conversational AI-based Assistant for Supporting the Human User in Detecting and Responding to Phishing Attacks**

cs.HC

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05951v1) [paper-pdf](http://arxiv.org/pdf/2502.05951v1)

**Authors**: Antonio La Torre, Marco Angelini

**Abstract**: This work introduces Cyri, an AI-powered conversational assistant designed to support a human user in detecting and analyzing phishing emails by leveraging Large Language Models. Cyri has been designed to scrutinize emails for semantic features used in phishing attacks, such as urgency, and undesirable consequences, using an approach that unifies features already established in the literature with others by Cyri features extraction methodology. Cyri can be directly plugged into a client mail or webmail, ensuring seamless integration with the user's email workflow while maintaining data privacy through local processing. By performing analyses on the user's machine, Cyri eliminates the need to transmit sensitive email data over the internet, reducing associated security risks. The Cyri user interface has been designed to reduce habituation effects and enhance user engagement. It employs dynamic visual cues and context-specific explanations to keep users alert and informed while using emails. Additionally, it allows users to explore identified malicious semantic features both through conversation with the agent and visual exploration, obtaining the advantages of both modalities for expert or non-expert users. It also allows users to keep track of the conversation, supports the user in solving additional questions on both computed features or new parts of the mail, and applies its detection on demand. To evaluate Cyri, we crafted a comprehensive dataset of 420 phishing emails and 420 legitimate emails. Results demonstrate high effectiveness in identifying critical phishing semantic features fundamental to phishing detection. A user study involving 10 participants, both experts and non-experts, evaluated Cyri's effectiveness and usability. Results indicated that Cyri significantly aided users in identifying phishing emails and enhanced their understanding of phishing tactics.



## **47. Large Language Models are Easily Confused: A Quantitative Metric, Security Implications and Typological Analysis**

cs.CL

18 pages, 15 figures, 14 tables

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.13237v2) [paper-pdf](http://arxiv.org/pdf/2410.13237v2)

**Authors**: Yiyi Chen, Qiongxiu Li, Russa Biswas, Johannes Bjerva

**Abstract**: Language Confusion is a phenomenon where Large Language Models (LLMs) generate text that is neither in the desired language, nor in a contextually appropriate language. This phenomenon presents a critical challenge in text generation by LLMs, often appearing as erratic and unpredictable behavior. We hypothesize that there are linguistic regularities to this inherent vulnerability in LLMs and shed light on patterns of language confusion across LLMs. We introduce a novel metric, Language Confusion Entropy, designed to directly measure and quantify this confusion, based on language distributions informed by linguistic typology and lexical variation. Comprehensive comparisons with the Language Confusion Benchmark (Marchisio et al., 2024) confirm the effectiveness of our metric, revealing patterns of language confusion across LLMs. We further link language confusion to LLM security, and find patterns in the case of multilingual embedding inversion attacks. Our analysis demonstrates that linguistic typology offers theoretically grounded interpretation, and valuable insights into leveraging language similarities as a prior for LLM alignment and security.



## **48. Arabic Dataset for LLM Safeguard Evaluation**

cs.CL

Accepted at NAACL 2025 Main Conference

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.17040v2) [paper-pdf](http://arxiv.org/pdf/2410.17040v2)

**Authors**: Yasser Ashraf, Yuxia Wang, Bin Gu, Preslav Nakov, Timothy Baldwin

**Abstract**: The growing use of large language models (LLMs) has raised concerns regarding their safety. While many studies have focused on English, the safety of LLMs in Arabic, with its linguistic and cultural complexities, remains under-explored. Here, we aim to bridge this gap. In particular, we present an Arab-region-specific safety evaluation dataset consisting of 5,799 questions, including direct attacks, indirect attacks, and harmless requests with sensitive words, adapted to reflect the socio-cultural context of the Arab world. To uncover the impact of different stances in handling sensitive and controversial topics, we propose a dual-perspective evaluation framework. It assesses the LLM responses from both governmental and opposition viewpoints. Experiments over five leading Arabic-centric and multilingual LLMs reveal substantial disparities in their safety performance. This reinforces the need for culturally specific datasets to ensure the responsible deployment of LLMs.



## **49. Mask-based Membership Inference Attacks for Retrieval-Augmented Generation**

cs.CR

This paper is accepted by conference WWW 2025

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.20142v2) [paper-pdf](http://arxiv.org/pdf/2410.20142v2)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long

**Abstract**: Retrieval-Augmented Generation (RAG) has been an effective approach to mitigate hallucinations in large language models (LLMs) by incorporating up-to-date and domain-specific knowledge. Recently, there has been a trend of storing up-to-date or copyrighted data in RAG knowledge databases instead of using it for LLM training. This practice has raised concerns about Membership Inference Attacks (MIAs), which aim to detect if a specific target document is stored in the RAG system's knowledge database so as to protect the rights of data producers. While research has focused on enhancing the trustworthiness of RAG systems, existing MIAs for RAG systems remain largely insufficient. Previous work either relies solely on the RAG system's judgment or is easily influenced by other documents or the LLM's internal knowledge, which is unreliable and lacks explainability. To address these limitations, we propose a Mask-Based Membership Inference Attacks (MBA) framework. Our framework first employs a masking algorithm that effectively masks a certain number of words in the target document. The masked text is then used to prompt the RAG system, and the RAG system is required to predict the mask values. If the target document appears in the knowledge database, the masked text will retrieve the complete target document as context, allowing for accurate mask prediction. Finally, we adopt a simple yet effective threshold-based method to infer the membership of target document by analyzing the accuracy of mask prediction. Our mask-based approach is more document-specific, making the RAG system's generation less susceptible to distractions from other documents or the LLM's internal knowledge. Extensive experiments demonstrate the effectiveness of our approach compared to existing baseline models.



## **50. Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails**

cs.CV

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05772v1) [paper-pdf](http://arxiv.org/pdf/2502.05772v1)

**Authors**: Yijun Yang, Lichao Wang, Xiao Yang, Lanqing Hong, Jun Zhu

**Abstract**: Vision Large Language Models (VLLMs) integrate visual data processing, expanding their real-world applications, but also increasing the risk of generating unsafe responses. In response, leading companies have implemented Multi-Layered safety defenses, including alignment training, safety system prompts, and content moderation. However, their effectiveness against sophisticated adversarial attacks remains largely unexplored. In this paper, we propose MultiFaceted Attack, a novel attack framework designed to systematically bypass Multi-Layered Defenses in VLLMs. It comprises three complementary attack facets: Visual Attack that exploits the multimodal nature of VLLMs to inject toxic system prompts through images; Alignment Breaking Attack that manipulates the model's alignment mechanism to prioritize the generation of contrasting responses; and Adversarial Signature that deceives content moderators by strategically placing misleading information at the end of the response. Extensive evaluations on eight commercial VLLMs in a black-box setting demonstrate that MultiFaceted Attack achieves a 61.56% attack success rate, surpassing state-of-the-art methods by at least 42.18%.



