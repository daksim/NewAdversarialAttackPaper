# Latest Large Language Model Attack Papers
**update at 2024-07-31 16:19:30**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Breaking Agents: Compromising Autonomous LLM Agents Through Malfunction Amplification**

cs.CR

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20859v1) [paper-pdf](http://arxiv.org/pdf/2407.20859v1)

**Authors**: Boyang Zhang, Yicong Tan, Yun Shen, Ahmed Salem, Michael Backes, Savvas Zannettou, Yang Zhang

**Abstract**: Recently, autonomous agents built on large language models (LLMs) have experienced significant development and are being deployed in real-world applications. These agents can extend the base LLM's capabilities in multiple ways. For example, a well-built agent using GPT-3.5-Turbo as its core can outperform the more advanced GPT-4 model by leveraging external components. More importantly, the usage of tools enables these systems to perform actions in the real world, moving from merely generating text to actively interacting with their environment. Given the agents' practical applications and their ability to execute consequential actions, it is crucial to assess potential vulnerabilities. Such autonomous systems can cause more severe damage than a standalone language model if compromised. While some existing research has explored harmful actions by LLM agents, our study approaches the vulnerability from a different perspective. We introduce a new type of attack that causes malfunctions by misleading the agent into executing repetitive or irrelevant actions. We conduct comprehensive evaluations using various attack methods, surfaces, and properties to pinpoint areas of susceptibility. Our experiments reveal that these attacks can induce failure rates exceeding 80\% in multiple scenarios. Through attacks on implemented and deployable agents in multi-agent scenarios, we accentuate the realistic risks associated with these vulnerabilities. To mitigate such attacks, we propose self-examination detection methods. However, our findings indicate these attacks are difficult to detect effectively using LLMs alone, highlighting the substantial risks associated with this vulnerability.



## **2. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

cs.CR

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.20361v1) [paper-pdf](http://arxiv.org/pdf/2407.20361v1)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of two existing models, Stack model and Phishpedia, in classifying PhishOracle-generated adversarial phishing webpages. Additionally, we study a commercial large language model, Gemini Pro Vision, in the context of adversarial attacks. We conduct a user study to determine whether PhishOracle-generated adversarial phishing webpages deceive users. Our findings reveal that many PhishOracle-generated phishing webpages evade current phishing webpage detection models and deceive users, but Gemini Pro Vision is robust to the attack. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources are publicly available on GitHub.



## **3. Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization**

cs.CL

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2406.00045v2) [paper-pdf](http://arxiv.org/pdf/2406.00045v2)

**Authors**: Yuanpu Cao, Tianrong Zhang, Bochuan Cao, Ziyi Yin, Lu Lin, Fenglong Ma, Jinghui Chen

**Abstract**: Researchers have been studying approaches to steer the behavior of Large Language Models (LLMs) and build personalized LLMs tailored for various applications. While fine-tuning seems to be a direct solution, it requires substantial computational resources and may significantly affect the utility of the original LLM. Recent endeavors have introduced more lightweight strategies, focusing on extracting "steering vectors" to guide the model's output toward desired behaviors by adjusting activations within specific layers of the LLM's transformer architecture. However, such steering vectors are directly extracted from the activations of human preference data and thus often lead to suboptimal results and occasional failures, especially in alignment-related scenarios. This work proposes an innovative approach that could produce more effective steering vectors through bi-directional preference optimization. Our method is designed to allow steering vectors to directly influence the generation probability of contrastive human preference data pairs, thereby offering a more precise representation of the target behavior. By carefully adjusting the direction and magnitude of the steering vector, we enabled personalized control over the desired behavior across a spectrum of intensities. Extensive experimentation across various open-ended generation tasks, particularly focusing on steering AI personas, has validated the efficacy of our approach. Moreover, we comprehensively investigate critical alignment-concerning scenarios, such as managing truthfulness, mitigating hallucination, and addressing jailbreaking attacks. Remarkably, our method can still demonstrate outstanding steering effectiveness across these scenarios. Furthermore, we showcase the transferability of our steering vectors across different models/LoRAs and highlight the synergistic benefits of applying multiple vectors simultaneously.



## **4. Can Editing LLMs Inject Harm?**

cs.CL

The first two authors contributed equally. 9 pages for main paper, 36  pages including appendix. The code, results, dataset for this paper and more  resources are on the project website: https://llm-editing.github.io

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.20224v1) [paper-pdf](http://arxiv.org/pdf/2407.20224v1)

**Authors**: Canyu Chen, Baixiang Huang, Zekun Li, Zhaorun Chen, Shiyang Lai, Xiongxiao Xu, Jia-Chen Gu, Jindong Gu, Huaxiu Yao, Chaowei Xiao, Xifeng Yan, William Yang Wang, Philip Torr, Dawn Song, Kai Shu

**Abstract**: Knowledge editing techniques have been increasingly adopted to efficiently correct the false or outdated knowledge in Large Language Models (LLMs), due to the high cost of retraining from scratch. Meanwhile, one critical but under-explored question is: can knowledge editing be used to inject harm into LLMs? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the risk of misinformation injection, we first categorize it into commonsense misinformation injection and long-tail misinformation injection. Then, we find that editing attacks can inject both types of misinformation into LLMs, and the effectiveness is particularly high for commonsense misinformation injection. For the risk of bias injection, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can cause a high bias increase in general outputs of LLMs, which are even highly irrelevant to the injected sentence, indicating a catastrophic impact on the overall fairness of LLMs. Then, we further illustrate the high stealthiness of editing attacks, measured by their impact on the general knowledge and reasoning capacities of LLMs, and show the hardness of defending editing attacks with empirical evidence. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs.



## **5. Large Language Models as Carriers of Hidden Messages**

cs.CL

Work in progress. Code is available at  https://github.com/j-hoscilowic/zurek-stegano

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2406.02481v2) [paper-pdf](http://arxiv.org/pdf/2406.02481v2)

**Authors**: Jakub Hoscilowicz, Pawel Popiolek, Jan Rudkowski, Jedrzej Bieniasz, Artur Janicki

**Abstract**: With the help of simple fine-tuning, one can artificially embed hidden text into large language models (LLMs). This text is revealed only when triggered by a specific query to the LLM. Two primary applications are LLM fingerprinting and steganography. In the context of LLM fingerprinting, a unique text identifier (fingerprint) is embedded within the model to verify licensing compliance. In the context of steganography, the LLM serves as a carrier for hidden messages that can be disclosed through a chosen trigger question.   Our work demonstrates that embedding hidden text in the LLM via fine-tuning, though seemingly secure due to the vast number of potential triggers (any sequence of characters or tokens could serve as a trigger), is susceptible to extraction through analysis of the LLM's output decoding process. We propose an extraction attack called Unconditional Token Forcing (UTF). It is premised on the hypothesis that iteratively feeding each token from the LLM's vocabulary into the model should reveal output sequences with abnormally high token probabilities, indicating potential hidden text candidates. We also present a defense method to hide text in such a way that it is resistant to both UTF and attacks based on sampling decoding methods, which we named Unconditional Token Forcing Confusion (UTFC). To the best of our knowledge, there is no attack method that can extract text hidden with UTFC. UTFC has both benign applications (improving LLM fingerprinting) and malign applications (using LLMs to create covert communication channels). Code is available at github.com/j-hoscilowic/zurek-stegano



## **6. Detecting and Understanding Vulnerabilities in Language Models via Mechanistic Interpretability**

cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.19842v1) [paper-pdf](http://arxiv.org/pdf/2407.19842v1)

**Authors**: Jorge García-Carrasco, Alejandro Maté, Juan Trujillo

**Abstract**: Large Language Models (LLMs), characterized by being trained on broad amounts of data in a self-supervised manner, have shown impressive performance across a wide range of tasks. Indeed, their generative abilities have aroused interest on the application of LLMs across a wide range of contexts. However, neural networks in general, and LLMs in particular, are known to be vulnerable to adversarial attacks, where an imperceptible change to the input can mislead the output of the model. This is a serious concern that impedes the use of LLMs on high-stakes applications, such as healthcare, where a wrong prediction can imply serious consequences. Even though there are many efforts on making LLMs more robust to adversarial attacks, there are almost no works that study \emph{how} and \emph{where} these vulnerabilities that make LLMs prone to adversarial attacks happen. Motivated by these facts, we explore how to localize and understand vulnerabilities, and propose a method, based on Mechanistic Interpretability (MI) techniques, to guide this process. Specifically, this method enables us to detect vulnerabilities related to a concrete task by (i) obtaining the subset of the model that is responsible for that task, (ii) generating adversarial samples for that task, and (iii) using MI techniques together with the previous samples to discover and understand the possible vulnerabilities. We showcase our method on a pretrained GPT-2 Small model carrying out the task of predicting 3-letter acronyms to demonstrate its effectiveness on locating and understanding concrete vulnerabilities of the model.



## **7. Fine-Tuning, Quantization, and LLMs: Navigating Unintended Outcomes**

cs.CR

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2404.04392v2) [paper-pdf](http://arxiv.org/pdf/2404.04392v2)

**Authors**: Divyanshu Kumar, Anurakt Kumar, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Large Language Models (LLMs) have become very popular and are used in many domains, such as chatbots, auto-task completion agents, and much more. However, LLMs suffer from many safety vulnerabilities, which can be exploited using different types of attacks, such as jailbreaking, prompt injection attacks, and privacy leakage attacks. These attacks can disrupt the working of the LLMs and make powerful LLM systems generate malicious or unethical content, take malicious actions, or leak confidential information by bypassing the security filters and taking advantage of their access. Foundational LLMs undergo alignment training, which includes safety training. This helps the model learn how to generate outputs that are ethical and aligned with human responses. Further, to make the models even safer, guardrails are added to filter the inputs received and the output generated by the model. These foundational LLMs are subjected to fine-tuning, quantization, or alteration of guardrails to use these models for specialized tasks or to use them in a resource-constrained environment. So, understanding the impact of modifications such as fine-tuning, quantization, and guardrails on the safety of LLM becomes an important question. Understanding and mitigating the consequences will help build reliable systems and effective strategies to make LLMs more secure. In this study, we tested foundational models like Mistral, Llama, MosaicML, and their finetuned versions. These comprehensive evaluations show that fine-tuning increases jailbreak attack success rates (ASR), quantization has a variable impact on the ASR, and guardrails can help significantly improve jailbreak resistance.



## **8. Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness**

cs.LG

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.12068v2) [paper-pdf](http://arxiv.org/pdf/2407.12068v2)

**Authors**: Kai Guo, Zewen Liu, Zhikai Chen, Hongzhi Wen, Wei Jin, Jiliang Tang, Yi Chang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.



## **9. Evolving Diverse Red-team Language Models in Multi-round Multi-agent Games**

cs.CL

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2310.00322v5) [paper-pdf](http://arxiv.org/pdf/2310.00322v5)

**Authors**: Chengdong Ma, Ziran Yang, Hai Ci, Jun Gao, Minquan Gao, Xuehai Pan, Yaodong Yang

**Abstract**: The primary challenge in deploying Large Language Model (LLM) is ensuring its harmlessness. Red team can identify vulnerabilities by attacking LLM to attain safety. However, current efforts heavily rely on single-round prompt designs and unilateral red team optimizations against fixed blue teams. These static approaches lead to significant reductions in generation diversity, known as the mode collapse, which makes it difficult to discover the potential risks in the increasingly complex human-LLM interactions. Here we introduce dynamic Red Team Game (RTG) to comprehensively analyze the multi-round offensive and defensive interactions between red team and blue team. Furthermore, we develop a Gamified Red Team Solver (GRTS) with diversity measures to mitigate mode collapse and theoretically guarantee the convergence of approximate Nash equilibrium which results in better strategies for both teams. Empirical results demonstrate that GRTS explore diverse and implicit attacks to adaptively exploit various LLMs, surpassing the constraints of specific modes. Insightfully, the geometrical structure we unveil of the red team task aligns with the spinning top hypothesis, confirming the necessity of constructing a diverse LLM population as a promising proxy for heterogeneous human expert red-teamers. This paves the way for scalable toxicity detection and safe alignment for LLMs.



## **10. Large Language Models for Cyber Security: A Systematic Literature Review**

cs.CR

47 pages,6 figures

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2405.04760v3) [paper-pdf](http://arxiv.org/pdf/2405.04760v3)

**Authors**: Hanxiang Xu, Shenao Wang, Ningke Li, Kailong Wang, Yanjie Zhao, Kai Chen, Ting Yu, Yang Liu, Haoyu Wang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened up new opportunities for leveraging artificial intelligence in various domains, including cybersecurity. As the volume and sophistication of cyber threats continue to grow, there is an increasing need for intelligent systems that can automatically detect vulnerabilities, analyze malware, and respond to attacks. In this survey, we conduct a comprehensive review of the literature on the application of LLMs in cybersecurity (LLM4Security). By comprehensively collecting over 30K relevant papers and systematically analyzing 127 papers from top security and software engineering venues, we aim to provide a holistic view of how LLMs are being used to solve diverse problems across the cybersecurity domain. Through our analysis, we identify several key findings. First, we observe that LLMs are being applied to a wide range of cybersecurity tasks, including vulnerability detection, malware analysis, network intrusion detection, and phishing detection. Second, we find that the datasets used for training and evaluating LLMs in these tasks are often limited in size and diversity, highlighting the need for more comprehensive and representative datasets. Third, we identify several promising techniques for adapting LLMs to specific cybersecurity domains, such as fine-tuning, transfer learning, and domain-specific pre-training. Finally, we discuss the main challenges and opportunities for future research in LLM4Security, including the need for more interpretable and explainable models, the importance of addressing data privacy and security concerns, and the potential for leveraging LLMs for proactive defense and threat hunting. Overall, our survey provides a comprehensive overview of the current state-of-the-art in LLM4Security and identifies several promising directions for future research.



## **11. LLM Platform Security: Applying a Systematic Evaluation Framework to OpenAI's ChatGPT Plugins**

cs.CR

To appear in the proceedings of the 7th AAAI / ACM Conference on AI,  Ethics, and Society (AIES), October 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2309.10254v2) [paper-pdf](http://arxiv.org/pdf/2309.10254v2)

**Authors**: Umar Iqbal, Tadayoshi Kohno, Franziska Roesner

**Abstract**: Large language model (LLM) platforms, such as ChatGPT, have recently begun offering an app ecosystem to interface with third-party services on the internet. While these apps extend the capabilities of LLM platforms, they are developed by arbitrary third parties and thus cannot be implicitly trusted. Apps also interface with LLM platforms and users using natural language, which can have imprecise interpretations. In this paper, we propose a framework that lays a foundation for LLM platform designers to analyze and improve the security, privacy, and safety of current and future third-party integrated LLM platforms. Our framework is a formulation of an attack taxonomy that is developed by iteratively exploring how LLM platform stakeholders could leverage their capabilities and responsibilities to mount attacks against each other. As part of our iterative process, we apply our framework in the context of OpenAI's plugin (apps) ecosystem. We uncover plugins that concretely demonstrate the potential for the types of issues that we outline in our attack taxonomy. We conclude by discussing novel challenges and by providing recommendations to improve the security, privacy, and safety of present and future LLM-based computing platforms.



## **12. Blockchain for Large Language Model Security and Safety: A Holistic Survey**

cs.CR

Submitted to SIGKDD Explorations

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.20181v1) [paper-pdf](http://arxiv.org/pdf/2407.20181v1)

**Authors**: Caleb Geren, Amanda Board, Gaby G. Dagher, Tim Andersen, Jun Zhuang

**Abstract**: With the advent of accessible interfaces for interacting with large language models, there has been an associated explosion in both their commercial and academic interest. Consequently, there has also been an sudden burst of novel attacks associated with large language models, jeopardizing user data on a massive scale. Situated at a comparable crossroads in its development, and equally prolific to LLMs in its rampant growth, blockchain has emerged in recent years as a disruptive technology with the potential to redefine how we approach data handling. In particular, and due to its strong guarantees about data immutability and irrefutability as well as inherent data provenance assurances, blockchain has attracted significant attention as a means to better defend against the array of attacks affecting LLMs and further improve the quality of their responses. In this survey, we holistically evaluate current research on how blockchains are being used to help protect against LLM vulnerabilities, as well as analyze how they may further be used in novel applications. To better serve these ends, we introduce a taxonomy of blockchain for large language models (BC4LLM) and also develop various definitions to precisely capture the nature of different bodies of research in these areas. Moreover, throughout the paper, we present frameworks to contextualize broader research efforts, and in order to motivate the field further, we identify future research goals as well as challenges present in the blockchain for large language model (BC4LLM) space.



## **13. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

cs.CR

This work has been accepted by CCS 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2310.15469v3) [paper-pdf](http://arxiv.org/pdf/2310.15469v3)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.



## **14. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

cs.CV

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2406.18849v2) [paper-pdf](http://arxiv.org/pdf/2406.18849v2)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.



## **15. MistralBSM: Leveraging Mistral-7B for Vehicular Networks Misbehavior Detection**

cs.LG

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18462v1) [paper-pdf](http://arxiv.org/pdf/2407.18462v1)

**Authors**: Wissal Hamhoum, Soumaya Cherkaoui

**Abstract**: Vehicular networks are exposed to various threats resulting from malicious attacks. These threats compromise the security and reliability of communications among road users, thereby jeopardizing road and traffic safety. One of the main vectors of these attacks within vehicular networks is misbehaving vehicles. To address this challenge, we propose deploying a pretrained Large Language Model (LLM)-empowered Misbehavior Detection System (MDS) within an edge-cloud detection framework. Specifically, we fine-tune Mistral-7B, a state-of-the-art LLM, as the edge component to enable real-time detection, whereas a larger LLM deployed in the cloud can conduct a more comprehensive analysis. Our experiments conducted on the extended VeReMi dataset demonstrate Mistral-7B's superior performance, achieving 98\% accuracy compared to other LLMs such as LLAMA2-7B and RoBERTa. Additionally, we investigate the impact of window size on computational costs to optimize deployment efficiency. Leveraging LLMs in MDS shows interesting results in improving the detection of vehicle misbehavior, consequently strengthening vehicular network security to ensure the safety of road users.



## **16. SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding**

cs.CR

To appear in ACL 2024

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2402.08983v4) [paper-pdf](http://arxiv.org/pdf/2402.08983v4)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, Radha Poovendran

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications such as code generation and chatbot assistance, extensive efforts have been made to align LLM behavior with human values, including safety. Jailbreak attacks, aiming to provoke unintended and unsafe behaviors from LLMs, remain a significant/leading LLM safety threat. In this paper, we aim to defend LLMs against jailbreak attacks by introducing SafeDecoding, a safety-aware decoding strategy for LLMs to generate helpful and harmless responses to user queries. Our insight in developing SafeDecoding is based on the observation that, even though probabilities of tokens representing harmful contents outweigh those representing harmless responses, safety disclaimers still appear among the top tokens after sorting tokens by probability in descending order. This allows us to mitigate jailbreak attacks by identifying safety disclaimers and amplifying their token probabilities, while simultaneously attenuating the probabilities of token sequences that are aligned with the objectives of jailbreak attacks. We perform extensive experiments on five LLMs using six state-of-the-art jailbreak attacks and four benchmark datasets. Our results show that SafeDecoding significantly reduces the attack success rate and harmfulness of jailbreak attacks without compromising the helpfulness of responses to benign user queries. SafeDecoding outperforms six defense methods.



## **17. Human-Interpretable Adversarial Prompt Attack on Large Language Models with Situational Context**

cs.CL

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.14644v2) [paper-pdf](http://arxiv.org/pdf/2407.14644v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous research on testing the vulnerabilities in Large Language Models (LLMs) using adversarial attacks has primarily focused on nonsensical prompt injections, which are easily detected upon manual or automated review (e.g., via byte entropy). However, the exploration of innocuous human-understandable malicious prompts augmented with adversarial injections remains limited. In this research, we explore converting a nonsensical suffix attack into a sensible prompt via a situation-driven contextual re-writing. This allows us to show suffix conversion without any gradients, using only LLMs to perform the attacks, and thus better understand the scope of possible risks. We combine an independent, meaningful adversarial insertion and situations derived from movies to check if this can trick an LLM. The situations are extracted from the IMDB dataset, and prompts are defined following a few-shot chain-of-thought prompting. Our approach demonstrates that a successful situation-driven attack can be executed on both open-source and proprietary LLMs. We find that across many LLMs, as few as 1 attempt produces an attack and that these attacks transfer between LLMs.



## **18. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2312.03853v4) [paper-pdf](http://arxiv.org/pdf/2312.03853v4)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Gemini (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Gemini. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.



## **19. The Dark Side of Function Calling: Pathways to Jailbreaking Large Language Models**

cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17915v1) [paper-pdf](http://arxiv.org/pdf/2407.17915v1)

**Authors**: Zihui Wu, Haichang Gao, Jianping He, Ping Wang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but their power comes with significant security considerations. While extensive research has been conducted on the safety of LLMs in chat mode, the security implications of their function calling feature have been largely overlooked. This paper uncovers a critical vulnerability in the function calling process of LLMs, introducing a novel "jailbreak function" attack method that exploits alignment discrepancies, user coercion, and the absence of rigorous safety filters. Our empirical study, conducted on six state-of-the-art LLMs including GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-pro, reveals an alarming average success rate of over 90\% for this attack. We provide a comprehensive analysis of why function calls are susceptible to such attacks and propose defensive strategies, including the use of defensive prompts. Our findings highlight the urgent need for enhanced security measures in the function calling capabilities of LLMs, contributing to the field of AI safety by identifying a previously unexplored risk, designing an effective attack method, and suggesting practical defensive measures. Our code is available at https://github.com/wooozihui/jailbreakfunction.



## **20. PenHeal: A Two-Stage LLM Framework for Automated Pentesting and Optimal Remediation**

cs.CR

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.17788v1) [paper-pdf](http://arxiv.org/pdf/2407.17788v1)

**Authors**: Junjie Huang, Quanyan Zhu

**Abstract**: Recent advances in Large Language Models (LLMs) have shown significant potential in enhancing cybersecurity defenses against sophisticated threats. LLM-based penetration testing is an essential step in automating system security evaluations by identifying vulnerabilities. Remediation, the subsequent crucial step, addresses these discovered vulnerabilities. Since details about vulnerabilities, exploitation methods, and software versions offer crucial insights into system weaknesses, integrating penetration testing with vulnerability remediation into a cohesive system has become both intuitive and necessary.   This paper introduces PenHeal, a two-stage LLM-based framework designed to autonomously identify and mitigate security vulnerabilities. The framework integrates two LLM-enabled components: the Pentest Module, which detects multiple vulnerabilities within a system, and the Remediation Module, which recommends optimal remediation strategies. The integration is facilitated through Counterfactual Prompting and an Instructor module that guides the LLMs using external knowledge to explore multiple potential attack paths effectively. Our experimental results demonstrate that PenHeal not only automates the identification and remediation of vulnerabilities but also significantly improves vulnerability coverage by 31%, increases the effectiveness of remediation strategies by 32%, and reduces the associated costs by 46% compared to baseline models. These outcomes highlight the transformative potential of LLMs in reshaping cybersecurity practices, offering an innovative solution to defend against cyber threats.



## **21. Towards Neural Network based Cognitive Models of Dynamic Decision-Making by Humans**

cs.LG

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17622v1) [paper-pdf](http://arxiv.org/pdf/2407.17622v1)

**Authors**: Changyu Chen, Shashank Reddy Chirra, Maria José Ferreira, Cleotilde Gonzalez, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Modelling human cognitive processes in dynamic decision-making tasks has been an endeavor in AI for a long time. Some initial works have attempted to utilize neural networks (and large language models) but often assume one common model for all humans and aim to emulate human behavior in aggregate. However, behavior of each human is distinct, heterogeneous and relies on specific past experiences in specific tasks. To that end, we build on a well known model of cognition, namely Instance Based Learning (IBL), that posits that decisions are made based on similar situations encountered in the past. We propose two new attention based neural network models to model human decision-making in dynamic settings. We experiment with two distinct datasets gathered from human subject experiment data, one focusing on detection of phishing email by humans and another where humans act as attackers in a cybersecurity setting and decide on an attack option. We conduct extensive experiments with our two neural network models, IBL, and GPT3.5, and demonstrate that one of our neural network models achieves the best performance in representing human decision-making. We find an interesting trend that all models predict a human's decision better if that human is better at the task. We also explore explanation of human decisions based on what our model considers important in prediction. Overall, our work yields promising results for further use of neural networks in cognitive modelling of human decision making. Our code is available at https://github.com/shshnkreddy/NCM-HDM.



## **22. Can Watermarking Large Language Models Prevent Copyrighted Text Generation and Hide Training Data?**

cs.LG

21 pages, 6 figures

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.17417v1) [paper-pdf](http://arxiv.org/pdf/2407.17417v1)

**Authors**: Michael-Andrei Panaitescu-Liess, Zora Che, Bang An, Yuancheng Xu, Pankayaraj Pathmanathan, Souradip Chakraborty, Sicheng Zhu, Tom Goldstein, Furong Huang

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in generating diverse and contextually rich text. However, concerns regarding copyright infringement arise as LLMs may inadvertently produce copyrighted material. In this paper, we first investigate the effectiveness of watermarking LLMs as a deterrent against the generation of copyrighted texts. Through theoretical analysis and empirical evaluation, we demonstrate that incorporating watermarks into LLMs significantly reduces the likelihood of generating copyrighted content, thereby addressing a critical concern in the deployment of LLMs. Additionally, we explore the impact of watermarking on Membership Inference Attacks (MIAs), which aim to discern whether a sample was part of the pretraining dataset and may be used to detect copyright violations. Surprisingly, we find that watermarking adversely affects the success rate of MIAs, complicating the task of detecting copyrighted text in the pretraining dataset. Finally, we propose an adaptive technique to improve the success rate of a recent MIA under watermarking. Our findings underscore the importance of developing adaptive methods to study critical problems in LLMs with potential legal implications.



## **23. LLMmap: Fingerprinting For Large Language Models**

cs.CR

version 0.1 (added missing refs)

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.15847v2) [paper-pdf](http://arxiv.org/pdf/2407.15847v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: We introduce LLMmap, a first-generation fingerprinting attack targeted at LLM-integrated applications. LLMmap employs an active fingerprinting approach, sending carefully crafted queries to the application and analyzing the responses to identify the specific LLM model in use. With as few as 8 interactions, LLMmap can accurately identify LLMs with over 95% accuracy. More importantly, LLMmap is designed to be robust across different application layers, allowing it to identify LLMs operating under various system prompts, stochastic sampling hyperparameters, and even complex generation frameworks such as RAG or Chain-of-Thought.



## **24. From Sands to Mansions: Enabling Automatic Full-Life-Cycle Cyberattack Construction with LLM**

cs.CR

**SubmitDate**: 2024-07-24    [abs](http://arxiv.org/abs/2407.16928v1) [paper-pdf](http://arxiv.org/pdf/2407.16928v1)

**Authors**: Lingzhi Wang, Jiahui Wang, Kyle Jung, Kedar Thiagarajan, Emily Wei, Xiangmin Shen, Yan Chen, Zhenyuan Li

**Abstract**: The escalating battles between attackers and defenders in cybersecurity make it imperative to test and evaluate defense capabilities from the attackers' perspective. However, constructing full-life-cycle cyberattacks and performing red team emulations requires significant time and domain knowledge from security experts. Existing cyberattack simulation frameworks face challenges such as limited technical coverage, inability to conduct full-life-cycle attacks, and the need for manual infrastructure building. These limitations hinder the quality and diversity of the constructed attacks. In this paper, we leveraged the capabilities of Large Language Models (LLMs) in summarizing knowledge from existing attack intelligence and generating executable machine code based on human knowledge. we proposed AURORA, an automatic end-to-end cyberattack construction and emulation framework. AURORA can autonomously build multi-stage cyberattack plans based on Cyber Threat Intelligence (CTI) reports, construct the emulation infrastructures, and execute the attack procedures. We also developed an attack procedure knowledge graph to integrate knowledge about attack techniques throughout the full life cycle of advanced cyberattacks from various sources. We constructed and evaluated more than 20 full-life-cycle cyberattacks based on existing CTI reports. Compared to previous attack simulation frameworks, AURORA can construct multi-step attacks and the infrastructures in several minutes without human intervention. Furthermore, AURORA incorporates a wider range (40% more) of attack techniques into the constructed attacks in a more efficient way than the professional red teams. To benefit further research, we open-sourced the dataset containing the execution files and infrastructures of 20 emulated cyberattacks.



## **25. RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**

cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2403.13031v2) [paper-pdf](http://arxiv.org/pdf/2403.13031v2)

**Authors**: Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, Bo Li

**Abstract**: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evaluations demonstrate that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks. The innovative use of constrained optimization and a fusion-based guardrail approach represents a significant step forward in developing more secure and reliable LLMs, setting a new standard for content moderation frameworks in the face of evolving digital threats.



## **26. Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models**

cs.LG

ICML 2024. Website: https://hanlin-zhang.com/impossibility-watermarks

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2311.04378v4) [paper-pdf](http://arxiv.org/pdf/2311.04378v4)

**Authors**: Hanlin Zhang, Benjamin L. Edelman, Danilo Francati, Daniele Venturi, Giuseppe Ateniese, Boaz Barak

**Abstract**: Watermarking generative models consists of planting a statistical signal (watermark) in a model's output so that it can be later verified that the output was generated by the given model. A strong watermarking scheme satisfies the property that a computationally bounded attacker cannot erase the watermark without causing significant quality degradation. In this paper, we study the (im)possibility of strong watermarking schemes. We prove that, under well-specified and natural assumptions, strong watermarking is impossible to achieve. This holds even in the private detection algorithm setting, where the watermark insertion and detection algorithms share a secret key, unknown to the attacker. To prove this result, we introduce a generic efficient watermark attack; the attacker is not required to know the private key of the scheme or even which scheme is used. Our attack is based on two assumptions: (1) The attacker has access to a "quality oracle" that can evaluate whether a candidate output is a high-quality response to a prompt, and (2) The attacker has access to a "perturbation oracle" which can modify an output with a nontrivial probability of maintaining quality, and which induces an efficiently mixing random walk on high-quality outputs. We argue that both assumptions can be satisfied in practice by an attacker with weaker computational capabilities than the watermarked model itself, to which the attacker has only black-box access. Furthermore, our assumptions will likely only be easier to satisfy over time as models grow in capabilities and modalities. We demonstrate the feasibility of our attack by instantiating it to attack three existing watermarking schemes for large language models: Kirchenbauer et al. (2023), Kuditipudi et al. (2023), and Zhao et al. (2023). The same attack successfully removes the watermarks planted by all three schemes, with only minor quality degradation.



## **27. Can Large Language Models Automatically Jailbreak GPT-4V?**

cs.CL

TrustNLP@NAACL2024 (Fourth Workshop on Trustworthy Natural Language  Processing)

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16686v1) [paper-pdf](http://arxiv.org/pdf/2407.16686v1)

**Authors**: Yuanwei Wu, Yue Huang, Yixin Liu, Xiang Li, Pan Zhou, Lichao Sun

**Abstract**: GPT-4V has attracted considerable attention due to its extraordinary capacity for integrating and processing multimodal information. At the same time, its ability of face recognition raises new safety concerns of privacy leakage. Despite researchers' efforts in safety alignment through RLHF or preprocessing filters, vulnerabilities might still be exploited. In our study, we introduce AutoJailbreak, an innovative automatic jailbreak technique inspired by prompt optimization. We leverage Large Language Models (LLMs) for red-teaming to refine the jailbreak prompt and employ weak-to-strong in-context learning prompts to boost efficiency. Furthermore, we present an effective search method that incorporates early stopping to minimize optimization time and token expenditure. Our experiments demonstrate that AutoJailbreak significantly surpasses conventional methods, achieving an Attack Success Rate (ASR) exceeding 95.3\%. This research sheds light on strengthening GPT-4V security, underscoring the potential for LLMs to be exploited in compromising GPT-4V integrity.



## **28. RedAgent: Red Teaming Large Language Models with Context-aware Autonomous Language Agent**

cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16667v1) [paper-pdf](http://arxiv.org/pdf/2407.16667v1)

**Authors**: Huiyu Xu, Wenhui Zhang, Zhibo Wang, Feng Xiao, Rui Zheng, Yunhe Feng, Zhongjie Ba, Kui Ren

**Abstract**: Recently, advanced Large Language Models (LLMs) such as GPT-4 have been integrated into many real-world applications like Code Copilot. These applications have significantly expanded the attack surface of LLMs, exposing them to a variety of threats. Among them, jailbreak attacks that induce toxic responses through jailbreak prompts have raised critical safety concerns. To identify these threats, a growing number of red teaming approaches simulate potential adversarial scenarios by crafting jailbreak prompts to test the target LLM. However, existing red teaming methods do not consider the unique vulnerabilities of LLM in different scenarios, making it difficult to adjust the jailbreak prompts to find context-specific vulnerabilities. Meanwhile, these methods are limited to refining jailbreak templates using a few mutation operations, lacking the automation and scalability to adapt to different scenarios. To enable context-aware and efficient red teaming, we abstract and model existing attacks into a coherent concept called "jailbreak strategy" and propose a multi-agent LLM system named RedAgent that leverages these strategies to generate context-aware jailbreak prompts. By self-reflecting on contextual feedback in an additional memory buffer, RedAgent continuously learns how to leverage these strategies to achieve effective jailbreaks in specific contexts. Extensive experiments demonstrate that our system can jailbreak most black-box LLMs in just five queries, improving the efficiency of existing red teaming methods by two times. Additionally, RedAgent can jailbreak customized LLM applications more efficiently. By generating context-aware jailbreak prompts towards applications on GPTs, we discover 60 severe vulnerabilities of these real-world applications with only two queries per vulnerability. We have reported all found issues and communicated with OpenAI and Meta for bug fixes.



## **29. Course-Correction: Safety Alignment Using Synthetic Preferences**

cs.CL

Dataset and script will be available at  https://github.com/pillowsofwind/Course-Correction

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16637v1) [paper-pdf](http://arxiv.org/pdf/2407.16637v1)

**Authors**: Rongwu Xu, Yishuo Cai, Zhenhong Zhou, Renjie Gu, Haiqin Weng, Yan Liu, Tianwei Zhang, Wei Xu, Han Qiu

**Abstract**: The risk of harmful content generated by large language models (LLMs) becomes a critical concern. This paper presents a systematic study on assessing and improving LLMs' capability to perform the task of \textbf{course-correction}, \ie, the model can steer away from generating harmful content autonomously. To start with, we introduce the \textsc{C$^2$-Eval} benchmark for quantitative assessment and analyze 10 popular LLMs, revealing varying proficiency of current safety-tuned LLMs in course-correction. To improve, we propose fine-tuning LLMs with preference learning, emphasizing the preference for timely course-correction. Using an automated pipeline, we create \textsc{C$^2$-Syn}, a synthetic dataset with 750K pairwise preferences, to teach models the concept of timely course-correction through data-driven preference learning. Experiments on 2 LLMs, \textsc{Llama2-Chat 7B} and \textsc{Qwen2 7B}, show that our method effectively enhances course-correction skills without affecting general performance. Additionally, it effectively improves LLMs' safety, particularly in resisting jailbreak attacks.



## **30. Prompt Injection Attacks on Large Language Models in Oncology**

cs.CR

57 Pages, 5 Figures

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.18981v1) [paper-pdf](http://arxiv.org/pdf/2407.18981v1)

**Authors**: Jan Clusmann, Dyke Ferber, Isabella C. Wiest, Carolin V. Schneider, Titus J. Brinker, Sebastian Foersch, Daniel Truhn, Jakob N. Kather

**Abstract**: Vision-language artificial intelligence models (VLMs) possess medical knowledge and can be employed in healthcare in numerous ways, including as image interpreters, virtual scribes, and general decision support systems. However, here, we demonstrate that current VLMs applied to medical tasks exhibit a fundamental security flaw: they can be attacked by prompt injection attacks, which can be used to output harmful information just by interacting with the VLM, without any access to its parameters. We performed a quantitative study to evaluate the vulnerabilities to these attacks in four state of the art VLMs which have been proposed to be of utility in healthcare: Claude 3 Opus, Claude 3.5 Sonnet, Reka Core, and GPT-4o. Using a set of N=297 attacks, we show that all of these models are susceptible. Specifically, we show that embedding sub-visual prompts in medical imaging data can cause the model to provide harmful output, and that these prompts are non-obvious to human observers. Thus, our study demonstrates a key vulnerability in medical VLMs which should be mitigated before widespread clinical adoption.



## **31. Defending Our Privacy With Backdoors**

cs.LG

Accepted at ECAI 2024

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2310.08320v4) [paper-pdf](http://arxiv.org/pdf/2310.08320v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information, such as names and faces of individuals, from vision-language models by fine-tuning them for only a few minutes instead of re-training them from scratch. Specifically, by strategically inserting backdoors into text encoders, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's actual name. For image encoders, we map individuals' embeddings to be removed from the model to a universal, anonymous embedding. The results of our extensive experimental evaluation demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides a new "dual-use" perspective on backdoor attacks and presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.



## **32. Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models**

cs.CV

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2404.10335v3) [paper-pdf](http://arxiv.org/pdf/2404.10335v3)

**Authors**: Qi Guo, Shanmin Pang, Xiaojun Jia, Yang Liu, Qing Guo

**Abstract**: Adversarial attacks, particularly \textbf{targeted} transfer-based attacks, can be used to assess the adversarial robustness of large visual-language models (VLMs), allowing for a more thorough examination of potential security flaws before deployment. However, previous transfer-based adversarial attacks incur high costs due to high iteration counts and complex method structure. Furthermore, due to the unnaturalness of adversarial semantics, the generated adversarial examples have low transferability. These issues limit the utility of existing methods for assessing robustness. To address these issues, we propose AdvDiffVLM, which uses diffusion models to generate natural, unrestricted and targeted adversarial examples via score matching. Specifically, AdvDiffVLM uses Adaptive Ensemble Gradient Estimation to modify the score during the diffusion model's reverse generation process, ensuring that the produced adversarial examples have natural adversarial targeted semantics, which improves their transferability. Simultaneously, to improve the quality of adversarial examples, we use the GradCAM-guided Mask method to disperse adversarial semantics throughout the image rather than concentrating them in a single area. Finally, AdvDiffVLM embeds more target semantics into adversarial examples after multiple iterations. Experimental results show that our method generates adversarial examples 5x to 10x faster than state-of-the-art transfer-based adversarial attacks while maintaining higher quality adversarial examples. Furthermore, compared to previous transfer-based adversarial attacks, the adversarial examples generated by our method have better transferability. Notably, AdvDiffVLM can successfully attack a variety of commercial VLMs in a black-box environment, including GPT-4V.



## **33. Figure it Out: Analyzing-based Jailbreak Attack on Large Language Models**

cs.CR

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16205v1) [paper-pdf](http://arxiv.org/pdf/2407.16205v1)

**Authors**: Shi Lin, Rongchang Li, Xun Wang, Changting Lin, Wenpeng Xing, Meng Han

**Abstract**: The rapid development of Large Language Models (LLMs) has brought remarkable generative capabilities across diverse tasks. However, despite the impressive achievements, these models still have numerous security vulnerabilities, particularly when faced with jailbreak attacks. Therefore, by investigating jailbreak attacks, we can uncover hidden weaknesses in LLMs and guide us in developing more robust defense mechanisms to fortify their security. In this paper, we further explore the boundary of jailbreak attacks on LLMs and propose Analyzing-based Jailbreak (ABJ). This effective jailbreak attack method takes advantage of LLMs' growing analyzing and reasoning capability and reveals their underlying vulnerabilities when facing analysis-based tasks. We conduct a detailed evaluation of ABJ across various open-source and closed-source LLMs, which achieves 94.8% Attack Success Rate (ASR) and 1.06 Attack Efficiency (AE) on GPT-4-turbo-0409, demonstrating state-of-the-art attack effectiveness and efficiency. Our research highlights the importance of prioritizing and enhancing the safety of LLMs to mitigate the risks of misuse.



## **34. Robust Privacy Amidst Innovation with Large Language Models Through a Critical Assessment of the Risks**

cs.CL

13 pages, 4 figures, 1 table, 1 supplementary, under review

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.16166v1) [paper-pdf](http://arxiv.org/pdf/2407.16166v1)

**Authors**: Yao-Shun Chuang, Atiquer Rahman Sarkar, Noman Mohammed, Xiaoqian Jiang

**Abstract**: This study examines integrating EHRs and NLP with large language models (LLMs) to improve healthcare data management and patient care. It focuses on using advanced models to create secure, HIPAA-compliant synthetic patient notes for biomedical research. The study used de-identified and re-identified MIMIC III datasets with GPT-3.5, GPT-4, and Mistral 7B to generate synthetic notes. Text generation employed templates and keyword extraction for contextually relevant notes, with one-shot generation for comparison. Privacy assessment checked PHI occurrence, while text utility was tested using an ICD-9 coding task. Text quality was evaluated with ROUGE and cosine similarity metrics to measure semantic similarity with source notes. Analysis of PHI occurrence and text utility via the ICD-9 coding task showed that the keyword-based method had low risk and good performance. One-shot generation showed the highest PHI exposure and PHI co-occurrence, especially in geographic location and date categories. The Normalized One-shot method achieved the highest classification accuracy. Privacy analysis revealed a critical balance between data utility and privacy protection, influencing future data use and sharing. Re-identified data consistently outperformed de-identified data. This study demonstrates the effectiveness of keyword-based methods in generating privacy-protecting synthetic clinical notes that retain data usability, potentially transforming clinical data-sharing practices. The superior performance of re-identified over de-identified data suggests a shift towards methods that enhance utility and privacy by using dummy PHIs to perplex privacy attacks.



## **35. Flooding Spread of Manipulated Knowledge in LLM-Based Multi-Agent Communities**

cs.CL

18 Pages, working in progress

**SubmitDate**: 2024-07-23    [abs](http://arxiv.org/abs/2407.07791v2) [paper-pdf](http://arxiv.org/pdf/2407.07791v2)

**Authors**: Tianjie Ju, Yiting Wang, Xinbei Ma, Pengzhou Cheng, Haodong Zhao, Yulong Wang, Lifeng Liu, Jian Xie, Zhuosheng Zhang, Gongshen Liu

**Abstract**: The rapid adoption of large language models (LLMs) in multi-agent systems has highlighted their impressive capabilities in various applications, such as collaborative problem-solving and autonomous negotiation. However, the security implications of these LLM-based multi-agent systems have not been thoroughly investigated, particularly concerning the spread of manipulated knowledge. In this paper, we investigate this critical issue by constructing a detailed threat model and a comprehensive simulation environment that mirrors real-world multi-agent deployments in a trusted platform. Subsequently, we propose a novel two-stage attack method involving Persuasiveness Injection and Manipulated Knowledge Injection to systematically explore the potential for manipulated knowledge (i.e., counterfactual and toxic knowledge) spread without explicit prompt manipulation.   Our method leverages the inherent vulnerabilities of LLMs in handling world knowledge, which can be exploited by attackers to unconsciously spread fabricated information. Through extensive experiments, we demonstrate that our attack method can successfully induce LLM-based agents to spread both counterfactual and toxic knowledge without degrading their foundational capabilities during agent communication. Furthermore, we show that these manipulations can persist through popular retrieval-augmented generation frameworks, where several benign agents store and retrieve manipulated chat histories for future interactions. This persistence indicates that even after the interaction has ended, the benign agents may continue to be influenced by manipulated knowledge. Our findings reveal significant security risks in LLM-based multi-agent systems, emphasizing the imperative need for robust defenses against manipulated knowledge spread, such as introducing ``guardian'' agents and advanced fact-checking tools.



## **36. The Shadow of Fraud: The Emerging Danger of AI-powered Social Engineering and its Possible Cure**

cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15912v1) [paper-pdf](http://arxiv.org/pdf/2407.15912v1)

**Authors**: Jingru Yu, Yi Yu, Xuhong Wang, Yilun Lin, Manzhi Yang, Yu Qiao, Fei-Yue Wang

**Abstract**: Social engineering (SE) attacks remain a significant threat to both individuals and organizations. The advancement of Artificial Intelligence (AI), including diffusion models and large language models (LLMs), has potentially intensified these threats by enabling more personalized and convincing attacks. This survey paper categorizes SE attack mechanisms, analyzes their evolution, and explores methods for measuring these threats. It highlights the challenges in raising awareness about the risks of AI-enhanced SE attacks and offers insights into developing proactive and adaptable defense strategies. Additionally, we introduce a categorization of the evolving nature of AI-powered social engineering attacks into "3E phases": Enlarging, wherein the magnitude of attacks expands through the leverage of digital media; Enriching, introducing novel attack vectors and techniques; and Emerging, signifying the advent of novel threats and methods. Moreover, we emphasize the necessity for a robust framework to assess the risk of AI-powered SE attacks. By identifying and addressing gaps in existing research, we aim to guide future studies and encourage the development of more effective defenses against the growing threat of AI-powered social engineering.



## **37. Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts**

cs.CL

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2402.16822v2) [paper-pdf](http://arxiv.org/pdf/2402.16822v2)

**Authors**: Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, Roberta Raileanu

**Abstract**: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to adversarial attacks is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel black-box approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem, and uses open-ended search to generate prompts that are both effective and diverse. Focusing on the safety domain, we use Rainbow Teaming to target various state-of-the-art LLMs, including the Llama 2 and Llama 3 models. Our approach reveals hundreds of effective adversarial prompts, with an attack success rate exceeding 90% across all tested models. Furthermore, we demonstrate that fine-tuning models with synthetic data generated by the Rainbow Teaming method significantly enhances their safety without sacrificing general performance or helpfulness. We additionally explore the versatility of Rainbow Teaming by applying it to question answering and cybersecurity, showcasing its potential to drive robust open-ended self-improvement in a wide range of applications.



## **38. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

cs.CL

8 pages

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2406.11260v2) [paper-pdf](http://arxiv.org/pdf/2406.11260v2)

**Authors**: Sungwon Park, Sungwon Han, Meeyoung Cha

**Abstract**: The spread of fake news negatively impacts individuals and is regarded as a significant social challenge that needs to be addressed. A number of algorithmic and insightful features have been identified for detecting fake news. However, with the recent LLMs and their advanced generation capabilities, many of the detectable features (e.g., style-conversion attacks) can be altered, making it more challenging to distinguish from real news. This study proposes adversarial style augmentation, AdStyle, to train a fake news detector that remains robust against various style-conversion attacks. Our model's key mechanism is the careful use of LLMs to automatically generate a diverse yet coherent range of style-conversion attack prompts. This improves the generation of prompts that are particularly difficult for the detector to handle. Experiments show that our augmentation strategy improves robustness and detection performance when tested on fake news benchmark datasets.



## **39. Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

cs.LG

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15549v1) [paper-pdf](http://arxiv.org/pdf/2407.15549v1)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of `jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.



## **40. TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs**

cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.09164v3) [paper-pdf](http://arxiv.org/pdf/2407.09164v3)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully used to simplify and facilitate code programming. With these tools, developers can easily generate desired complete functional codes based on incomplete code and natural language prompts. However, a few pioneering works revealed that these Code LLMs are also vulnerable, e.g., against backdoor and adversarial attacks. The former could induce LLMs to respond to triggers to insert malicious code snippets by poisoning the training data or model parameters, while the latter can craft malicious adversarial input codes to reduce the quality of generated codes. However, both attack methods have underlying limitations: backdoor attacks rely on controlling the model training process, while adversarial attacks struggle with fulfilling specific malicious purposes.   To inherit the advantages of both backdoor and adversarial attacks, this paper proposes a new attack paradigm, i.e., target-specific and adversarial prompt injection (TAPI), against Code LLMs. TAPI generates unreadable comments containing information about malicious instructions and hides them as triggers in the external source code. When users exploit Code LLMs to complete codes containing the trigger, the models will generate attacker-specified malicious code snippets at specific locations. We evaluate our TAPI attack on four representative LLMs under three representative malicious objectives and seven cases. The results show that our method is highly threatening (achieving an attack success rate of up to 98.3%) and stealthy (saving an average of 53.1% of tokens in the trigger design). In particular, we successfully attack some famous deployed code completion integrated applications, including CodeGeex and Github Copilot. This further confirms the realistic threat of our attack.



## **41. Imposter.AI: Adversarial Attacks with Hidden Intentions towards Aligned Large Language Models**

cs.CL

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2407.15399v1) [paper-pdf](http://arxiv.org/pdf/2407.15399v1)

**Authors**: Xiao Liu, Liangzhi Li, Tong Xiang, Fuying Ye, Lu Wei, Wangyue Li, Noa Garcia

**Abstract**: With the development of large language models (LLMs) like ChatGPT, both their vast applications and potential vulnerabilities have come to the forefront. While developers have integrated multiple safety mechanisms to mitigate their misuse, a risk remains, particularly when models encounter adversarial inputs. This study unveils an attack mechanism that capitalizes on human conversation strategies to extract harmful information from LLMs. We delineate three pivotal strategies: (i) decomposing malicious questions into seemingly innocent sub-questions; (ii) rewriting overtly malicious questions into more covert, benign-sounding ones; (iii) enhancing the harmfulness of responses by prompting models for illustrative examples. Unlike conventional methods that target explicit malicious responses, our approach delves deeper into the nature of the information provided in responses. Through our experiments conducted on GPT-3.5-turbo, GPT-4, and Llama2, our method has demonstrated a marked efficacy compared to conventional attack methods. In summary, this work introduces a novel attack method that outperforms previous approaches, raising an important question: How to discern whether the ultimate intent in a dialogue is malicious?



## **42. Advancing TTP Analysis: Harnessing the Power of Large Language Models with Retrieval Augmented Generation**

cs.CR

**SubmitDate**: 2024-07-22    [abs](http://arxiv.org/abs/2401.00280v3) [paper-pdf](http://arxiv.org/pdf/2401.00280v3)

**Authors**: Reza Fayyazi, Rozhina Taghdimi, Shanchieh Jay Yang

**Abstract**: Tactics, Techniques, and Procedures (TTPs) outline the methods attackers use to exploit vulnerabilities. The interpretation of TTPs in the MITRE ATT&CK framework can be challenging for cybersecurity practitioners due to presumed expertise and complex dependencies. Meanwhile, advancements with Large Language Models (LLMs) have led to recent surge in studies exploring its uses in cybersecurity operations. It is, however, unclear how LLMs can be used in an efficient and proper way to provide accurate responses for critical domains such as cybersecurity. This leads us to investigate how to better use two types of LLMs: small-scale encoder-only (e.g., RoBERTa) and larger decoder-only (e.g., GPT-3.5) LLMs to comprehend and summarize TTPs with the intended purposes (i.e., tactics) of a cyberattack procedure. This work studies and compares the uses of supervised fine-tuning (SFT) of encoder-only LLMs vs. Retrieval Augmented Generation (RAG) for decoder-only LLMs (without fine-tuning). Both SFT and RAG techniques presumably enhance the LLMs with relevant contexts for each cyberattack procedure. Our studies show decoder-only LLMs with RAG achieves better performance than encoder-only models with SFT, particularly when directly relevant context is extracted by RAG. The decoder-only results could suffer low `Precision' while achieving high `Recall'. Our findings further highlight a counter-intuitive observation that more generic prompts tend to yield better predictions of cyberattack tactics than those that are more specifically tailored.



## **43. When Do Universal Image Jailbreaks Transfer Between Vision-Language Models?**

cs.CL

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.15211v1) [paper-pdf](http://arxiv.org/pdf/2407.15211v1)

**Authors**: Rylan Schaeffer, Dan Valentine, Luke Bailey, James Chua, Cristóbal Eyzaguirre, Zane Durante, Joe Benton, Brando Miranda, Henry Sleight, John Hughes, Rajashree Agrawal, Mrinank Sharma, Scott Emmons, Sanmi Koyejo, Ethan Perez

**Abstract**: The integration of new modalities into frontier AI systems offers exciting capabilities, but also increases the possibility such systems can be adversarially manipulated in undesirable ways. In this work, we focus on a popular class of vision-language models (VLMs) that generate text outputs conditioned on visual and textual inputs. We conducted a large-scale empirical study to assess the transferability of gradient-based universal image "jailbreaks" using a diverse set of over 40 open-parameter VLMs, including 18 new VLMs that we publicly release. Overall, we find that transferable gradient-based image jailbreaks are extremely difficult to obtain. When an image jailbreak is optimized against a single VLM or against an ensemble of VLMs, the jailbreak successfully jailbreaks the attacked VLM(s), but exhibits little-to-no transfer to any other VLMs; transfer is not affected by whether the attacked and target VLMs possess matching vision backbones or language models, whether the language model underwent instruction-following and/or safety-alignment training, or many other factors. Only two settings display partially successful transfer: between identically-pretrained and identically-initialized VLMs with slightly different VLM training data, and between different training checkpoints of a single VLM. Leveraging these results, we then demonstrate that transfer can be significantly improved against a specific target VLM by attacking larger ensembles of "highly-similar" VLMs. These results stand in stark contrast to existing evidence of universal and transferable text jailbreaks against language models and transferable adversarial attacks against image classifiers, suggesting that VLMs may be more robust to gradient-based transfer attacks.



## **44. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

cs.LG

jumps-Diffusion and stock market: Better quantify uncertainty in  financial simulations

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.14573v1) [paper-pdf](http://arxiv.org/pdf/2407.14573v1)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.



## **45. Arondight: Red Teaming Large Vision Language Models with Auto-generated Multi-modal Jailbreak Prompts**

cs.LG

To be published in ACM MM 2024

**SubmitDate**: 2024-07-21    [abs](http://arxiv.org/abs/2407.15050v1) [paper-pdf](http://arxiv.org/pdf/2407.15050v1)

**Authors**: Yi Liu, Chengjun Cai, Xiaoli Zhang, Xingliang Yuan, Cong Wang

**Abstract**: Large Vision Language Models (VLMs) extend and enhance the perceptual abilities of Large Language Models (LLMs). Despite offering new possibilities for LLM applications, these advancements raise significant security and ethical concerns, particularly regarding the generation of harmful content. While LLMs have undergone extensive security evaluations with the aid of red teaming frameworks, VLMs currently lack a well-developed one. To fill this gap, we introduce Arondight, a standardized red team framework tailored specifically for VLMs. Arondight is dedicated to resolving issues related to the absence of visual modality and inadequate diversity encountered when transitioning existing red teaming methodologies from LLMs to VLMs. Our framework features an automated multi-modal jailbreak attack, wherein visual jailbreak prompts are produced by a red team VLM, and textual prompts are generated by a red team LLM guided by a reinforcement learning agent. To enhance the comprehensiveness of VLM security evaluation, we integrate entropy bonuses and novelty reward metrics. These elements incentivize the RL agent to guide the red team LLM in creating a wider array of diverse and previously unseen test cases. Our evaluation of ten cutting-edge VLMs exposes significant security vulnerabilities, particularly in generating toxic images and aligning multi-modal prompts. In particular, our Arondight achieves an average attack success rate of 84.5\% on GPT-4 in all fourteen prohibited scenarios defined by OpenAI in terms of generating toxic text. For a clearer comparison, we also categorize existing VLMs based on their safety levels and provide corresponding reinforcement recommendations. Our multimodal prompt dataset and red team code will be released after ethics committee approval. CONTENT WARNING: THIS PAPER CONTAINS HARMFUL MODEL RESPONSES.



## **46. Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models**

cs.CV

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14971v1) [paper-pdf](http://arxiv.org/pdf/2407.14971v1)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Vision-language models (VLMs) have achieved significant strides in recent times specially in multimodal tasks, yet they remain susceptible to adversarial attacks on their vision components. To address this, we propose Sim-CLIP, an unsupervised adversarial fine-tuning method that enhances the robustness of the widely-used CLIP vision encoder against such attacks while maintaining semantic richness and specificity. By employing a Siamese architecture with cosine similarity loss, Sim-CLIP learns semantically meaningful and attack-resilient visual representations without requiring large batch sizes or momentum encoders. Our results demonstrate that VLMs enhanced with Sim-CLIP's fine-tuned CLIP encoder exhibit significantly enhanced robustness against adversarial attacks, while preserving semantic meaning of the perturbed images. Notably, Sim-CLIP does not require additional training or fine-tuning of the VLM itself; replacing the original vision encoder with our fine-tuned Sim-CLIP suffices to provide robustness. This work underscores the significance of reinforcing foundational models like CLIP to safeguard the reliability of downstream VLM applications, paving the way for more secure and effective multimodal systems.



## **47. Operationalizing a Threat Model for Red-Teaming Large Language Models (LLMs)**

cs.CL

Preprint. Under review

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14937v1) [paper-pdf](http://arxiv.org/pdf/2407.14937v1)

**Authors**: Apurv Verma, Satyapriya Krishna, Sebastian Gehrmann, Madhavan Seshadri, Anu Pradhan, Tom Ault, Leslie Barrett, David Rabinowitz, John Doucette, NhatHai Phan

**Abstract**: Creating secure and resilient applications with large language models (LLM) requires anticipating, adjusting to, and countering unforeseen threats. Red-teaming has emerged as a critical technique for identifying vulnerabilities in real-world LLM implementations. This paper presents a detailed threat model and provides a systematization of knowledge (SoK) of red-teaming attacks on LLMs. We develop a taxonomy of attacks based on the stages of the LLM development and deployment process and extract various insights from previous research. In addition, we compile methods for defense and practical red-teaming strategies for practitioners. By delineating prominent attack motifs and shedding light on various entry points, this paper provides a framework for improving the security and robustness of LLM-based systems.



## **48. DistillSeq: A Framework for Safety Alignment Testing in Large Language Models using Knowledge Distillation**

cs.SE

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.10106v3) [paper-pdf](http://arxiv.org/pdf/2407.10106v3)

**Authors**: Mingke Yang, Yuqi Chen, Yi Liu, Ling Shi

**Abstract**: Large Language Models (LLMs) have showcased their remarkable capabilities in diverse domains, encompassing natural language understanding, translation, and even code generation. The potential for LLMs to generate harmful content is a significant concern. This risk necessitates rigorous testing and comprehensive evaluation of LLMs to ensure safe and responsible use. However, extensive testing of LLMs requires substantial computational resources, making it an expensive endeavor. Therefore, exploring cost-saving strategies during the testing phase is crucial to balance the need for thorough evaluation with the constraints of resource availability. To address this, our approach begins by transferring the moderation knowledge from an LLM to a small model. Subsequently, we deploy two distinct strategies for generating malicious queries: one based on a syntax tree approach, and the other leveraging an LLM-based method. Finally, our approach incorporates a sequential filter-test process designed to identify test cases that are prone to eliciting toxic responses. Our research evaluated the efficacy of DistillSeq across four LLMs: GPT-3.5, GPT-4.0, Vicuna-13B, and Llama-13B. In the absence of DistillSeq, the observed attack success rates on these LLMs stood at 31.5% for GPT-3.5, 21.4% for GPT-4.0, 28.3% for Vicuna-13B, and 30.9% for Llama-13B. However, upon the application of DistillSeq, these success rates notably increased to 58.5%, 50.7%, 52.5%, and 54.4%, respectively. This translated to an average escalation in attack success rate by a factor of 93.0% when compared to scenarios without the use of DistillSeq. Such findings highlight the significant enhancement DistillSeq offers in terms of reducing the time and resource investment required for effectively testing LLMs.



## **49. Retrieval Augmented Generation Integrated Large Language Models in Smart Contract Vulnerability Detection**

cs.CR

17 pages, 3 figures, 4 tables

**SubmitDate**: 2024-07-20    [abs](http://arxiv.org/abs/2407.14838v1) [paper-pdf](http://arxiv.org/pdf/2407.14838v1)

**Authors**: Jeffy Yu

**Abstract**: The rapid growth of Decentralized Finance (DeFi) has been accompanied by substantial financial losses due to smart contract vulnerabilities, underscoring the critical need for effective security auditing. With attacks becoming more frequent, the necessity and demand for auditing services has escalated. This especially creates a financial burden for independent developers and small businesses, who often have limited available funding for these services. Our study builds upon existing frameworks by integrating Retrieval-Augmented Generation (RAG) with large language models (LLMs), specifically employing GPT-4-1106 for its 128k token context window. We construct a vector store of 830 known vulnerable contracts, leveraging Pinecone for vector storage, OpenAI's text-embedding-ada-002 for embeddings, and LangChain to construct the RAG-LLM pipeline. Prompts were designed to provide a binary answer for vulnerability detection. We first test 52 smart contracts 40 times each against a provided vulnerability type, verifying the replicability and consistency of the RAG-LLM. Encouraging results were observed, with a 62.7% success rate in guided detection of vulnerabilities. Second, we challenge the model under a "blind" audit setup, without the vulnerability type provided in the prompt, wherein 219 contracts undergo 40 tests each. This setup evaluates the general vulnerability detection capabilities without hinted context assistance. Under these conditions, a 60.71% success rate was observed. While the results are promising, we still emphasize the need for human auditing at this time. We provide this study as a proof of concept for a cost-effective smart contract auditing process, moving towards democratic access to security.



## **50. CVE-LLM : Automatic vulnerability evaluation in medical device industry using large language models**

cs.CL

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.14640v1) [paper-pdf](http://arxiv.org/pdf/2407.14640v1)

**Authors**: Rikhiya Ghosh, Oladimeji Farri, Hans-Martin von Stockhausen, Martin Schmitt, George Marica Vasile

**Abstract**: The healthcare industry is currently experiencing an unprecedented wave of cybersecurity attacks, impacting millions of individuals. With the discovery of thousands of vulnerabilities each month, there is a pressing need to drive the automation of vulnerability assessment processes for medical devices, facilitating rapid mitigation efforts. Generative AI systems have revolutionized various industries, offering unparalleled opportunities for automation and increased efficiency. This paper presents a solution leveraging Large Language Models (LLMs) to learn from historical evaluations of vulnerabilities for the automatic assessment of vulnerabilities in the medical devices industry. This approach is applied within the portfolio of a single manufacturer, taking into account device characteristics, including existing security posture and controls. The primary contributions of this paper are threefold. Firstly, it provides a detailed examination of the best practices for training a vulnerability Language Model (LM) in an industrial context. Secondly, it presents a comprehensive comparison and insightful analysis of the effectiveness of Language Models in vulnerability assessment. Finally, it proposes a new human-in-the-loop framework to expedite vulnerability evaluation processes.



