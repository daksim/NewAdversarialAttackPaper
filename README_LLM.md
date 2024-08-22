# Latest Large Language Model Attack Papers
**update at 2024-08-22 17:17:14**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Competence-Based Analysis of Language Models**

cs.CL

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2303.00333v4) [paper-pdf](http://arxiv.org/pdf/2303.00333v4)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent successes of large, pretrained neural language models (LLMs), comparatively little is known about the representations of linguistic structure they learn during pretraining, which can lead to unexpected behaviors in response to prompt variation or distribution shift. To better understand these models and behaviors, we introduce a general model analysis framework to study LLMs with respect to their representation and use of human-interpretable linguistic properties. Our framework, CALM (Competence-based Analysis of Language Models), is designed to investigate LLM competence in the context of specific tasks by intervening on models' internal representations of different linguistic properties using causal probing, and measuring models' alignment under these interventions with a given ground-truth causal model of the task. We also develop a new approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than prior techniques. Finally, we carry out a case study of CALM using these interventions to analyze and compare LLM competence across a variety of lexical inference tasks, showing that CALM can be used to explain and predict behaviors across these tasks.



## **2. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11749v1) [paper-pdf](http://arxiv.org/pdf/2408.11749v1)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.



## **3. Watch Out for Your Guidance on Generation! Exploring Conditional Backdoor Attacks against Large Language Models**

cs.CL

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2404.14795v4) [paper-pdf](http://arxiv.org/pdf/2404.14795v4)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream backdoor attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of backdoor activation, we present a new poisoning paradigm against LLMs triggered by specifying generation conditions, which are commonly adopted strategies by users during model inference. The poisoned model performs normally for output under normal/other generation conditions, while becomes harmful for output under target generation conditions. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation conditions by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our attack can be generally divided into two types with different targets: Safety unalignment attack and Ability degradation attack. Our extensive experiments demonstrate that BrieFool is effective across safety domains and ability domains, achieving higher success rates than baseline methods, with 94.3 % on GPT-3.5-turbo



## **4. Large Language Models are Good Attackers: Efficient and Stealthy Textual Backdoor Attacks**

cs.CL

Under Review

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11587v1) [paper-pdf](http://arxiv.org/pdf/2408.11587v1)

**Authors**: Ziqiang Li, Yueqi Zeng, Pengfei Xia, Lei Liu, Zhangjie Fu, Bin Li

**Abstract**: With the burgeoning advancements in the field of natural language processing (NLP), the demand for training data has increased significantly. To save costs, it has become common for users and businesses to outsource the labor-intensive task of data collection to third-party entities. Unfortunately, recent research has unveiled the inherent risk associated with this practice, particularly in exposing NLP systems to potential backdoor attacks. Specifically, these attacks enable malicious control over the behavior of a trained model by poisoning a small portion of the training data. Unlike backdoor attacks in computer vision, textual backdoor attacks impose stringent requirements for attack stealthiness. However, existing attack methods meet significant trade-off between effectiveness and stealthiness, largely due to the high information entropy inherent in textual data. In this paper, we introduce the Efficient and Stealthy Textual backdoor attack method, EST-Bad, leveraging Large Language Models (LLMs). Our EST-Bad encompasses three core strategies: optimizing the inherent flaw of models as the trigger, stealthily injecting triggers with LLMs, and meticulously selecting the most impactful samples for backdoor injection. Through the integration of these techniques, EST-Bad demonstrates an efficient achievement of competitive attack performance while maintaining superior stealthiness compared to prior methods across various text classifier datasets.



## **5. SHIELD: Evaluation and Defense Strategies for Copyright Compliance in LLM Text Generation**

cs.CL

Work in progress

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2406.12975v2) [paper-pdf](http://arxiv.org/pdf/2406.12975v2)

**Authors**: Xiaoze Liu, Ting Sun, Tianyang Xu, Feijie Wu, Cunxiang Wang, Xiaoqian Wang, Jing Gao

**Abstract**: Large Language Models (LLMs) have transformed machine learning but raised significant legal concerns due to their potential to produce text that infringes on copyrights, resulting in several high-profile lawsuits. The legal landscape is struggling to keep pace with these rapid advancements, with ongoing debates about whether generated text might plagiarize copyrighted materials. Current LLMs may infringe on copyrights or overly restrict non-copyrighted texts, leading to these challenges: (i) the need for a comprehensive evaluation benchmark to assess copyright compliance from multiple aspects; (ii) evaluating robustness against safeguard bypassing attacks; and (iii) developing effective defense targeted against the generation of copyrighted text. To tackle these challenges, we introduce a curated dataset to evaluate methods, test attack strategies, and propose lightweight, real-time defense to prevent the generation of copyrighted text, ensuring the safe and lawful use of LLMs. Our experiments demonstrate that current LLMs frequently output copyrighted text, and that jailbreaking attacks can significantly increase the volume of copyrighted output. Our proposed defense mechanism significantly reduces the volume of copyrighted text generated by LLMs by effectively refusing malicious requests. Code is publicly available at https://github.com/xz-liu/SHIELD



## **6. Probing the Safety Response Boundary of Large Language Models via Unsafe Decoding Path Generation**

cs.CR

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.10668v2) [paper-pdf](http://arxiv.org/pdf/2408.10668v2)

**Authors**: Haoyu Wang, Bingzhe Wu, Yatao Bian, Yongzhe Chang, Xueqian Wang, Peilin Zhao

**Abstract**: Large Language Models (LLMs) are implicit troublemakers. While they provide valuable insights and assist in problem-solving, they can also potentially serve as a resource for malicious activities. Implementing safety alignment could mitigate the risk of LLMs generating harmful responses. We argue that: even when an LLM appears to successfully block harmful queries, there may still be hidden vulnerabilities that could act as ticking time bombs. To identify these underlying weaknesses, we propose to use a cost value model as both a detector and an attacker. Trained on external or self-generated harmful datasets, the cost value model could successfully influence the original safe LLM to output toxic content in decoding process. For instance, LLaMA-2-chat 7B outputs 39.18% concrete toxic content, along with only 22.16% refusals without any harmful suffixes. These potential weaknesses can then be exploited via prompt optimization such as soft prompts on images. We name this decoding strategy: Jailbreak Value Decoding (JVD), emphasizing that seemingly secure LLMs may not be as safe as we initially believe. They could be used to gather harmful data or launch covert attacks.



## **7. EEG-Defender: Defending against Jailbreak through Early Exit Generation of Large Language Models**

cs.AI

19 pages, 7 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11308v1) [paper-pdf](http://arxiv.org/pdf/2408.11308v1)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. Built upon this idea, we introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85\% in comparison with 50\% for the present SOTAs, with minimal impact on the utility and effectiveness of LLMs.



## **8. Medical MLLM is Vulnerable: Cross-Modality Jailbreak and Mismatched Attacks on Medical Multimodal Large Language Models**

cs.CR

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2405.20775v2) [paper-pdf](http://arxiv.org/pdf/2405.20775v2)

**Authors**: Xijie Huang, Xinyuan Wang, Hantao Zhang, Yinghao Zhu, Jiawen Xi, Jingkun An, Hao Wang, Hao Liang, Chengwei Pan

**Abstract**: Security concerns related to Large Language Models (LLMs) have been extensively explored, yet the safety implications for Multimodal Large Language Models (MLLMs), particularly in medical contexts (MedMLLMs), remain insufficiently studied. This paper delves into the underexplored security vulnerabilities of MedMLLMs, especially when deployed in clinical environments where the accuracy and relevance of question-and-answer interactions are critically tested against complex medical challenges. By combining existing clinical medical data with atypical natural phenomena, we define the mismatched malicious attack (2M-attack) and introduce its optimized version, known as the optimized mismatched malicious attack (O2M-attack or 2M-optimization). Using the voluminous 3MAD dataset that we construct, which covers a wide range of medical image modalities and harmful medical scenarios, we conduct a comprehensive analysis and propose the MCM optimization method, which significantly enhances the attack success rate on MedMLLMs. Evaluations with this dataset and attack methods, including white-box attacks on LLaVA-Med and transfer attacks (black-box) on four other SOTA models, indicate that even MedMLLMs designed with enhanced security features remain vulnerable to security breaches. Our work underscores the urgent need for a concerted effort to implement robust security measures and enhance the safety and efficacy of open-source MedMLLMs, particularly given the potential severity of jailbreak attacks and other malicious or clinically significant exploits in medical settings. Our code is available at https://github.com/dirtycomputer/O2M_attack.



## **9. Hide Your Malicious Goal Into Benign Narratives: Jailbreak Large Language Models through Neural Carrier Articles**

cs.CR

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.11182v1) [paper-pdf](http://arxiv.org/pdf/2408.11182v1)

**Authors**: Zhilong Wang, Haizhou Wang, Nanqing Luo, Lan Zhang, Xiaoyan Sun, Yebo Cao, Peng Liu

**Abstract**: Jailbreak attacks on Language Model Models (LLMs) entail crafting prompts aimed at exploiting the models to generate malicious content. This paper proposes a new type of jailbreak attacks which shift the attention of the LLM by inserting a prohibited query into a carrier article. The proposed attack leverage the knowledge graph and a composer LLM to automatically generating a carrier article that is similar to the topic of the prohibited query but does not violate LLM's safeguards. By inserting the malicious query to the carrier article, the assembled attack payload can successfully jailbreak LLM. To evaluate the effectiveness of our method, we leverage 4 popular categories of ``harmful behaviors'' adopted by related researches to attack 6 popular LLMs. Our experiment results show that the proposed attacking method can successfully jailbreak all the target LLMs which high success rate, except for Claude-3.



## **10. Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks**

cs.CL

Accepted to KDD 2024 (Research Track)

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2310.10830v2) [paper-pdf](http://arxiv.org/pdf/2310.10830v2)

**Authors**: Jiaying Wu, Jiafeng Guo, Bryan Hooi

**Abstract**: It is commonly perceived that fake news and real news exhibit distinct writing styles, such as the use of sensationalist versus objective language. However, we emphasize that style-related features can also be exploited for style-based attacks. Notably, the advent of powerful Large Language Models (LLMs) has empowered malicious actors to mimic the style of trustworthy news sources, doing so swiftly, cost-effectively, and at scale. Our analysis reveals that LLM-camouflaged fake news content significantly undermines the effectiveness of state-of-the-art text-based detectors (up to 38% decrease in F1 Score), implying a severe vulnerability to stylistic variations. To address this, we introduce SheepDog, a style-robust fake news detector that prioritizes content over style in determining news veracity. SheepDog achieves this resilience through (1) LLM-empowered news reframings that inject style diversity into the training process by customizing articles to match different styles; (2) a style-agnostic training scheme that ensures consistent veracity predictions across style-diverse reframings; and (3) content-focused veracity attributions that distill content-centric guidelines from LLMs for debunking fake news, offering supplementary cues and potential intepretability that assist veracity prediction. Extensive experiments on three real-world benchmarks demonstrate SheepDog's style robustness and adaptability to various backbones.



## **11. While GitHub Copilot Excels at Coding, Does It Ensure Responsible Output?**

cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.11006v1) [paper-pdf](http://arxiv.org/pdf/2408.11006v1)

**Authors**: Wen Cheng, Ke Sun, Xinyu Zhang, Wei Wang

**Abstract**: The rapid development of large language models (LLMs) has significantly advanced code completion capabilities, giving rise to a new generation of LLM-based Code Completion Tools (LCCTs). Unlike general-purpose LLMs, these tools possess unique workflows, integrating multiple information sources as input and prioritizing code suggestions over natural language interaction, which introduces distinct security challenges. Additionally, LCCTs often rely on proprietary code datasets for training, raising concerns about the potential exposure of sensitive data. This paper exploits these distinct characteristics of LCCTs to develop targeted attack methodologies on two critical security risks: jailbreaking and training data extraction attacks. Our experimental results expose significant vulnerabilities within LCCTs, including a 99.4% success rate in jailbreaking attacks on GitHub Copilot and a 46.3% success rate on Amazon Q. Furthermore, We successfully extracted sensitive user data from GitHub Copilot, including 54 real email addresses and 314 physical addresses associated with GitHub usernames. Our study also demonstrates that these code-based attack methods are effective against general-purpose LLMs, such as the GPT series, highlighting a broader security misalignment in the handling of code by modern LLMs. These findings underscore critical security challenges associated with LCCTs and suggest essential directions for strengthening their security frameworks. The example code and attack samples from our research are provided at https://github.com/Sensente/Security-Attacks-on-LCCTs.



## **12. Towards Efficient Formal Verification of Spiking Neural Network**

cs.AI

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10900v1) [paper-pdf](http://arxiv.org/pdf/2408.10900v1)

**Authors**: Baekryun Seong, Jieung Kim, Sang-Ki Ko

**Abstract**: Recently, AI research has primarily focused on large language models (LLMs), and increasing accuracy often involves scaling up and consuming more power. The power consumption of AI has become a significant societal issue; in this context, spiking neural networks (SNNs) offer a promising solution. SNNs operate event-driven, like the human brain, and compress information temporally. These characteristics allow SNNs to significantly reduce power consumption compared to perceptron-based artificial neural networks (ANNs), highlighting them as a next-generation neural network technology. However, societal concerns regarding AI go beyond power consumption, with the reliability of AI models being a global issue. For instance, adversarial attacks on AI models are a well-studied problem in the context of traditional neural networks. Despite their importance, the stability and property verification of SNNs remains in the early stages of research. Most SNN verification methods are time-consuming and barely scalable, making practical applications challenging. In this paper, we introduce temporal encoding to achieve practical performance in verifying the adversarial robustness of SNNs. We conduct a theoretical analysis of this approach and demonstrate its success in verifying SNNs at previously unmanageable scales. Our contribution advances SNN verification to a practical level, facilitating the safer application of SNNs.



## **13. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

cs.CR

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10738v1) [paper-pdf](http://arxiv.org/pdf/2408.10738v1)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also encounter notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the top k relevant items from offline knowledge bases, utilizing all available information from a webpage, including logos, HTML, and URLs. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.



## **14. MEGen: Generative Backdoor in Large Language Models via Model Editing**

cs.CL

Working in progress

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10722v1) [paper-pdf](http://arxiv.org/pdf/2408.10722v1)

**Authors**: Jiyang Qiu, Xinbei Ma, Zhuosheng Zhang, Hai Zhao

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities. Their powerful generative abilities enable flexible responses based on various queries or instructions. Emerging as widely adopted generalists for diverse tasks, LLMs are still vulnerable to backdoors. This paper proposes an editing-based generative backdoor, named MEGen, aiming to create a customized backdoor for NLP tasks with the least side effects. In our approach, we first leverage a language model to insert a trigger selected on fixed metrics into the input, then design a pipeline of model editing to directly embed a backdoor into an LLM. By adjusting a small set of local parameters with a mini-batch of samples, MEGen significantly enhances time efficiency and achieves high robustness. Experimental results indicate that our backdoor attack strategy achieves a high attack success rate on poison data while maintaining the model's performance on clean data. Notably, the backdoored model, when triggered, can freely output pre-set dangerous information while successfully completing downstream tasks. This suggests that future LLM applications could be guided to deliver certain dangerous information, thus altering the LLM's generative style. We believe this approach provides insights for future LLM applications and the execution of backdoor attacks on conversational AI systems.



## **15. Ferret: Faster and Effective Automated Red Teaming with Reward-Based Scoring Technique**

cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10701v1) [paper-pdf](http://arxiv.org/pdf/2408.10701v1)

**Authors**: Tej Deep Pala, Vernon Y. H. Toh, Rishabh Bhardwaj, Soujanya Poria

**Abstract**: In today's era, where large language models (LLMs) are integrated into numerous real-world applications, ensuring their safety and robustness is crucial for responsible AI usage. Automated red-teaming methods play a key role in this process by generating adversarial attacks to identify and mitigate potential vulnerabilities in these models. However, existing methods often struggle with slow performance, limited categorical diversity, and high resource demands. While Rainbow Teaming, a recent approach, addresses the diversity challenge by framing adversarial prompt generation as a quality-diversity search, it remains slow and requires a large fine-tuned mutator for optimal performance. To overcome these limitations, we propose Ferret, a novel approach that builds upon Rainbow Teaming by generating multiple adversarial prompt mutations per iteration and using a scoring function to rank and select the most effective adversarial prompt. We explore various scoring functions, including reward models, Llama Guard, and LLM-as-a-judge, to rank adversarial mutations based on their potential harm to improve the efficiency of the search for harmful mutations. Our results demonstrate that Ferret, utilizing a reward model as a scoring function, improves the overall attack success rate (ASR) to 95%, which is 46% higher than Rainbow Teaming. Additionally, Ferret reduces the time needed to achieve a 90% ASR by 15.2% compared to the baseline and generates adversarial prompts that are transferable i.e. effective on other LLMs of larger size. Our codes are available at https://github.com/declare-lab/ferret.



## **16. Promoting Equality in Large Language Models: Identifying and Mitigating the Implicit Bias based on Bayesian Theory**

cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10608v1) [paper-pdf](http://arxiv.org/pdf/2408.10608v1)

**Authors**: Yongxin Deng, Xihe Qiu, Xiaoyu Tan, Jing Pan, Chen Jue, Zhijun Fang, Yinghui Xu, Wei Chu, Yuan Qi

**Abstract**: Large language models (LLMs) are trained on extensive text corpora, which inevitably include biased information. Although techniques such as Affective Alignment can mitigate some negative impacts of these biases, existing prompt-based attack methods can still extract these biases from the model's weights. Moreover, these biases frequently appear subtly when LLMs are prompted to perform identical tasks across different demographic groups, thereby camouflaging their presence. To address this issue, we have formally defined the implicit bias problem and developed an innovative framework for bias removal based on Bayesian theory, Bayesian-Theory based Bias Removal (BTBR). BTBR employs likelihood ratio screening to pinpoint data entries within publicly accessible biased datasets that represent biases inadvertently incorporated during the LLM training phase. It then automatically constructs relevant knowledge triples and expunges bias information from LLMs using model editing techniques. Through extensive experimentation, we have confirmed the presence of the implicit bias problem in LLMs and demonstrated the effectiveness of our BTBR approach.



## **17. PromptBench: A Unified Library for Evaluation of Large Language Models**

cs.AI

Accepted by Journal of Machine Learning Research (JMLR); code:  https://github.com/microsoft/promptbench

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2312.07910v3) [paper-pdf](http://arxiv.org/pdf/2312.07910v3)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.



## **18. Development of an AI Anti-Bullying System Using Large Language Model Key Topic Detection**

cs.AI

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10417v1) [paper-pdf](http://arxiv.org/pdf/2408.10417v1)

**Authors**: Matthew Tassava, Cameron Kolodjski, Jordan Milbrath, Adorah Bishop, Nathan Flanders, Robbie Fetsch, Danielle Hanson, Jeremy Straub

**Abstract**: This paper presents and evaluates work on the development of an artificial intelligence (AI) anti-bullying system. The system is designed to identify coordinated bullying attacks via social media and other mechanisms, characterize them and propose remediation and response activities to them. In particular, a large language model (LLM) is used to populate an enhanced expert system-based network model of a bullying attack. This facilitates analysis and remediation activity - such as generating report messages to social media companies - determination. The system is described and the efficacy of the LLM for populating the model is analyzed herein.



## **19. A Disguised Wolf Is More Harmful Than a Toothless Tiger: Adaptive Malicious Code Injection Backdoor Attack Leveraging User Behavior as Triggers**

cs.AI

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10334v1) [paper-pdf](http://arxiv.org/pdf/2408.10334v1)

**Authors**: Shangxi Wu, Jitao Sang

**Abstract**: In recent years, large language models (LLMs) have made significant progress in the field of code generation. However, as more and more users rely on these models for software development, the security risks associated with code generation models have become increasingly significant. Studies have shown that traditional deep learning robustness issues also negatively impact the field of code generation. In this paper, we first present the game-theoretic model that focuses on security issues in code generation scenarios. This framework outlines possible scenarios and patterns where attackers could spread malicious code models to create security threats. We also pointed out for the first time that the attackers can use backdoor attacks to dynamically adjust the timing of malicious code injection, which will release varying degrees of malicious code depending on the skill level of the user. Through extensive experiments on leading code generation models, we validate our proposed game-theoretic model and highlight the significant threats that these new attack scenarios pose to the safe use of code models.



## **20. Topic-Based Watermarks for LLM-Generated Text**

cs.CR

Results for proposed scheme, additional/removal of content (figures  and equations), 12 pages

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2404.02138v3) [paper-pdf](http://arxiv.org/pdf/2404.02138v3)

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday

**Abstract**: The indistinguishability of text generated by large language models (LLMs) from human-generated text poses significant challenges. Watermarking algorithms are potential solutions by embedding detectable signatures within LLM-generated outputs. However, current watermarking schemes lack robustness to a range of attacks such as text substitution or manipulation, undermining their reliability. This paper proposes a novel topic-based watermarking algorithm for LLMs, designed to enhance the robustness of watermarking in LLMs. Our approach leverages the topics extracted from input prompts or outputs of non-watermarked LLMs in the generation process of watermarked text. We dynamically utilize token lists on identified topics and adjust token sampling weights accordingly. By using these topic-specific token biases, we embed a topic-sensitive watermarking into the generated text. We outline the theoretical framework of our topic-based watermarking algorithm and discuss its potential advantages in various scenarios. Additionally, we explore a comprehensive range of attacks against watermarking algorithms, including discrete alterations, paraphrasing, and tokenizations. We demonstrate that our proposed watermarking scheme classifies various watermarked text topics with 99.99% confidence and outperforms existing algorithms in terms of z-score robustness and the feasibility of modeling text degradation by potential attackers, while considering the trade-offs between the benefits and losses of watermarking LLM-generated text.



## **21. Privacy Checklist: Privacy Violation Detection Grounding on Contextual Integrity Theory**

cs.CL

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10053v1) [paper-pdf](http://arxiv.org/pdf/2408.10053v1)

**Authors**: Haoran Li, Wei Fan, Yulin Chen, Jiayang Cheng, Tianshu Chu, Xuebing Zhou, Peizhao Hu, Yangqiu Song

**Abstract**: Privacy research has attracted wide attention as individuals worry that their private data can be easily leaked during interactions with smart devices, social platforms, and AI applications. Computer science researchers, on the other hand, commonly study privacy issues through privacy attacks and defenses on segmented fields. Privacy research is conducted on various sub-fields, including Computer Vision (CV), Natural Language Processing (NLP), and Computer Networks. Within each field, privacy has its own formulation. Though pioneering works on attacks and defenses reveal sensitive privacy issues, they are narrowly trapped and cannot fully cover people's actual privacy concerns. Consequently, the research on general and human-centric privacy research remains rather unexplored. In this paper, we formulate the privacy issue as a reasoning problem rather than simple pattern matching. We ground on the Contextual Integrity (CI) theory which posits that people's perceptions of privacy are highly correlated with the corresponding social context. Based on such an assumption, we develop the first comprehensive checklist that covers social identities, private attributes, and existing privacy regulations. Unlike prior works on CI that either cover limited expert annotated norms or model incomplete social context, our proposed privacy checklist uses the whole Health Insurance Portability and Accountability Act of 1996 (HIPAA) as an example, to show that we can resort to large language models (LLMs) to completely cover the HIPAA's regulations. Additionally, our checklist also gathers expert annotations across multiple ontologies to determine private information including but not limited to personally identifiable information (PII). We use our preliminary results on the HIPAA to shed light on future context-centric privacy research to cover more privacy regulations, social norms and standards.



## **22. Transferring Backdoors between Large Language Models by Knowledge Distillation**

cs.CR

13 pages, 16 figures, 5 tables

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.09878v1) [paper-pdf](http://arxiv.org/pdf/2408.09878v1)

**Authors**: Pengzhou Cheng, Zongru Wu, Tianjie Ju, Wei Du, Zhuosheng Zhang Gongshen Liu

**Abstract**: Backdoor Attacks have been a serious vulnerability against Large Language Models (LLMs). However, previous methods only reveal such risk in specific models, or present tasks transferability after attacking the pre-trained phase. So, how risky is the model transferability of a backdoor attack? In this paper, we focus on whether existing mini-LLMs may be unconsciously instructed in backdoor knowledge by poisoned teacher LLMs through knowledge distillation (KD). Specifically, we propose ATBA, an adaptive transferable backdoor attack, which can effectively distill the backdoor of teacher LLMs into small models when only executing clean-tuning. We first propose the Target Trigger Generation (TTG) module that filters out a set of indicative trigger candidates from the token list based on cosine similarity distribution. Then, we exploit a shadow model to imitate the distilling process and introduce an Adaptive Trigger Optimization (ATO) module to realize a gradient-based greedy feedback to search optimal triggers. Extensive experiments show that ATBA generates not only positive guidance for student models but also implicitly transfers backdoor knowledge. Our attack is robust and stealthy, with over 80% backdoor transferability, and hopes the attention of security.



## **23. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

cs.CR

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2312.00029v3) [paper-pdf](http://arxiv.org/pdf/2312.00029v3)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. Such vulnerabilities can lead to LLMs being manipulated into generating hazardous content: from instructions for creating dangerous materials to inciting violence or endorsing unethical behaviors. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM acting as a guardian to the primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis reviews that by using Bergeron to complement models with existing alignment training, we can significantly improve the robustness and safety of multiple, commonly used commercial and open-source LLMs. Specifically, we found that models integrated with Bergeron are, on average, nearly seven times more resistant to attacks compared to models without such support.



## **24. Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning**

cs.AI

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09600v1) [paper-pdf](http://arxiv.org/pdf/2408.09600v1)

**Authors**: Tiansheng Huang, Gautam Bhattacharya, Pratik Joshi, Josh Kimball, Ling Liu

**Abstract**: Safety aligned Large Language Models (LLMs) are vulnerable to harmful fine-tuning attacks \cite{qi2023fine}-- a few harmful data mixed in the fine-tuning dataset can break the LLMs's safety alignment. Existing mitigation strategies include alignment stage solutions \cite{huang2024vaccine, rosati2024representation} and fine-tuning stage solutions \cite{huang2024lazy,mukhoti2023fine}. However, our evaluation shows that both categories of defenses fail \textit{when some specific training hyper-parameters are chosen} -- a large learning rate or a large number of training epochs in the fine-tuning stage can easily invalidate the defense, which however, is necessary to guarantee finetune performance. To this end, we propose Antidote, a post-fine-tuning stage solution, which remains \textbf{\textit{agnostic to the training hyper-parameters in the fine-tuning stage}}. Antidote relies on the philosophy that by removing the harmful parameters, the harmful model can be recovered from the harmful behaviors, regardless of how those harmful parameters are formed in the fine-tuning stage. With this philosophy, we introduce a one-shot pruning stage after harmful fine-tuning to remove the harmful weights that are responsible for the generation of harmful content. Despite its embarrassing simplicity, empirical results show that Antidote can reduce harmful score while maintaining accuracy on downstream tasks.



## **25. Characterizing and Evaluating the Reliability of LLMs against Jailbreak Attacks**

cs.CL

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09326v1) [paper-pdf](http://arxiv.org/pdf/2408.09326v1)

**Authors**: Kexin Chen, Yi Liu, Dongxia Wang, Jiaying Chen, Wenhai Wang

**Abstract**: Large Language Models (LLMs) have increasingly become pivotal in content generation with notable societal impact. These models hold the potential to generate content that could be deemed harmful.Efforts to mitigate this risk include implementing safeguards to ensure LLMs adhere to social ethics.However, despite such measures, the phenomenon of "jailbreaking" -- where carefully crafted prompts elicit harmful responses from models -- persists as a significant challenge. Recognizing the continuous threat posed by jailbreaking tactics and their repercussions for the trustworthy use of LLMs, a rigorous assessment of the models' robustness against such attacks is essential. This study introduces an comprehensive evaluation framework and conducts an large-scale empirical experiment to address this need. We concentrate on 10 cutting-edge jailbreak strategies across three categories, 1525 questions from 61 specific harmful categories, and 13 popular LLMs. We adopt multi-dimensional metrics such as Attack Success Rate (ASR), Toxicity Score, Fluency, Token Length, and Grammatical Errors to thoroughly assess the LLMs' outputs under jailbreak. By normalizing and aggregating these metrics, we present a detailed reliability score for different LLMs, coupled with strategic recommendations to reduce their susceptibility to such vulnerabilities. Additionally, we explore the relationships among the models, attack strategies, and types of harmful content, as well as the correlations between the evaluation metrics, which proves the validity of our multifaceted evaluation framework. Our extensive experimental results demonstrate a lack of resilience among all tested LLMs against certain strategies, and highlight the need to concentrate on the reliability facets of LLMs. We believe our study can provide valuable insights into enhancing the security evaluation of LLMs against jailbreak within the domain.



## **26. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09093v1) [paper-pdf](http://arxiv.org/pdf/2408.09093v1)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **27. Can Editing LLMs Inject Harm?**

cs.CL

The first two authors contributed equally. 9 pages for main paper, 36  pages including appendix. The code, results, dataset for this paper and more  resources are on the project website: https://llm-editing.github.io

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2407.20224v3) [paper-pdf](http://arxiv.org/pdf/2407.20224v3)

**Authors**: Canyu Chen, Baixiang Huang, Zekun Li, Zhaorun Chen, Shiyang Lai, Xiongxiao Xu, Jia-Chen Gu, Jindong Gu, Huaxiu Yao, Chaowei Xiao, Xifeng Yan, William Yang Wang, Philip Torr, Dawn Song, Kai Shu

**Abstract**: Knowledge editing has been increasingly adopted to correct the false or outdated knowledge in Large Language Models (LLMs). Meanwhile, one critical but under-explored question is: can knowledge editing be used to inject harm into LLMs? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the risk of misinformation injection, we first categorize it into commonsense misinformation injection and long-tail misinformation injection. Then, we find that editing attacks can inject both types of misinformation into LLMs, and the effectiveness is particularly high for commonsense misinformation injection. For the risk of bias injection, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can cause a bias increase in general outputs of LLMs, which are even highly irrelevant to the injected sentence, indicating a catastrophic impact on the overall fairness of LLMs. Then, we further illustrate the high stealthiness of editing attacks, measured by their impact on the general knowledge and reasoning capacities of LLMs, and show the hardness of defending editing attacks with empirical evidence. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs and the feasibility of disseminating misinformation or bias with LLMs as new channels.



## **28. Ask, Attend, Attack: A Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models**

cs.AI

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08989v1) [paper-pdf](http://arxiv.org/pdf/2408.08989v1)

**Authors**: Qingyuan Zeng, Zhenzhong Wang, Yiu-ming Cheung, Min Jiang

**Abstract**: While image-to-text models have demonstrated significant advancements in various vision-language tasks, they remain susceptible to adversarial attacks. Existing white-box attacks on image-to-text models require access to the architecture, gradients, and parameters of the target model, resulting in low practicality. Although the recently proposed gray-box attacks have improved practicality, they suffer from semantic loss during the training process, which limits their targeted attack performance. To advance adversarial attacks of image-to-text models, this paper focuses on a challenging scenario: decision-based black-box targeted attacks where the attackers only have access to the final output text and aim to perform targeted attacks. Specifically, we formulate the decision-based black-box targeted attack as a large-scale optimization problem. To efficiently solve the optimization problem, a three-stage process \textit{Ask, Attend, Attack}, called \textit{AAA}, is proposed to coordinate with the solver. \textit{Ask} guides attackers to create target texts that satisfy the specific semantics. \textit{Attend} identifies the crucial regions of the image for attacking, thus reducing the search space for the subsequent \textit{Attack}. \textit{Attack} uses an evolutionary algorithm to attack the crucial regions, where the attacks are semantically related to the target texts of \textit{Ask}, thus achieving targeted attacks without semantic loss. Experimental results on transformer-based and CNN+RNN-based image-to-text models confirmed the effectiveness of our proposed \textit{AAA}.



## **29. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

cs.LG

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08685v1) [paper-pdf](http://arxiv.org/pdf/2408.08685v1)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial perturbations, especially for topology attacks, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attack. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph.



## **30. MIA-Tuner: Adapting Large Language Models as Pre-training Text Detector**

cs.CL

code and dataset: https://github.com/wjfu99/MIA-Tuner

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08661v1) [paper-pdf](http://arxiv.org/pdf/2408.08661v1)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: The increasing parameters and expansive dataset of large language models (LLMs) highlight the urgent demand for a technical solution to audit the underlying privacy risks and copyright issues associated with LLMs. Existing studies have partially addressed this need through an exploration of the pre-training data detection problem, which is an instance of a membership inference attack (MIA). This problem involves determining whether a given piece of text has been used during the pre-training phase of the target LLM. Although existing methods have designed various sophisticated MIA score functions to achieve considerable detection performance in pre-trained LLMs, how to achieve high-confidence detection and how to perform MIA on aligned LLMs remain challenging. In this paper, we propose MIA-Tuner, a novel instruction-based MIA method, which instructs LLMs themselves to serve as a more precise pre-training data detector internally, rather than design an external MIA score function. Furthermore, we design two instruction-based safeguards to respectively mitigate the privacy risks brought by the existing methods and MIA-Tuner. To comprehensively evaluate the most recent state-of-the-art LLMs, we collect a more up-to-date MIA benchmark dataset, named WIKIMIA-24, to replace the widely adopted benchmark WIKIMIA. We conduct extensive experiments across various aligned and unaligned LLMs over the two benchmark datasets. The results demonstrate that MIA-Tuner increases the AUC of MIAs from 0.7 to a significantly high level of 0.9.



## **31. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

cs.IR

Survey paper

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2407.06992v2) [paper-pdf](http://arxiv.org/pdf/2407.06992v2)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.



## **32. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

cs.LG

END (will never be modified again) :Jumps-Diffusion and stock market:  Better quantify uncertainty in financial simulations

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2407.14573v3) [paper-pdf](http://arxiv.org/pdf/2407.14573v3)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.



## **33. ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages**

cs.CL

Accepted by ACL 2024 Main Conference

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2402.10753v2) [paper-pdf](http://arxiv.org/pdf/2402.10753v2)

**Authors**: Junjie Ye, Sixian Li, Guanyu Li, Caishuang Huang, Songyang Gao, Yilong Wu, Qi Zhang, Tao Gui, Xuanjing Huang

**Abstract**: Tool learning is widely acknowledged as a foundational approach or deploying large language models (LLMs) in real-world scenarios. While current research primarily emphasizes leveraging tools to augment LLMs, it frequently neglects emerging safety considerations tied to their application. To fill this gap, we present *ToolSword*, a comprehensive framework dedicated to meticulously investigating safety issues linked to LLMs in tool learning. Specifically, ToolSword delineates six safety scenarios for LLMs in tool learning, encompassing **malicious queries** and **jailbreak attacks** in the input stage, **noisy misdirection** and **risky cues** in the execution stage, and **harmful feedback** and **error conflicts** in the output stage. Experiments conducted on 11 open-source and closed-source LLMs reveal enduring safety challenges in tool learning, such as handling harmful queries, employing risky tools, and delivering detrimental feedback, which even GPT-4 is susceptible to. Moreover, we conduct further studies with the aim of fostering research on tool learning safety. The data is released in https://github.com/Junjie-Ye/ToolSword.



## **34. \textit{MMJ-Bench}: A Comprehensive Study on Jailbreak Attacks and Defenses for Vision Language Models**

cs.CR

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08464v1) [paper-pdf](http://arxiv.org/pdf/2408.08464v1)

**Authors**: Fenghua Weng, Yue Xu, Chengyan Fu, Wenjie Wang

**Abstract**: As deep learning advances, Large Language Models (LLMs) and their multimodal counterparts, Vision-Language Models (VLMs), have shown exceptional performance in many real-world tasks. However, VLMs face significant security challenges, such as jailbreak attacks, where attackers attempt to bypass the model's safety alignment to elicit harmful responses. The threat of jailbreak attacks on VLMs arises from both the inherent vulnerabilities of LLMs and the multiple information channels that VLMs process. While various attacks and defenses have been proposed, there is a notable gap in unified and comprehensive evaluations, as each method is evaluated on different dataset and metrics, making it impossible to compare the effectiveness of each method. To address this gap, we introduce \textit{MMJ-Bench}, a unified pipeline for evaluating jailbreak attacks and defense techniques for VLMs. Through extensive experiments, we assess the effectiveness of various attack methods against SoTA VLMs and evaluate the impact of defense mechanisms on both defense effectiveness and model utility for normal tasks. Our comprehensive evaluation contribute to the field by offering a unified and systematic evaluation framework and the first public-available benchmark for VLM jailbreak research. We also demonstrate several insightful findings that highlights directions for future studies.



## **35. Trojan Activation Attack: Red-Teaming Large Language Models using Activation Steering for Safety-Alignment**

cs.CR

ACM International Conference on Information and Knowledge Management  (CIKM'24)

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2311.09433v3) [paper-pdf](http://arxiv.org/pdf/2311.09433v3)

**Authors**: Haoran Wang, Kai Shu

**Abstract**: To ensure AI safety, instruction-tuned Large Language Models (LLMs) are specifically trained to ensure alignment, which refers to making models behave in accordance with human intentions. While these models have demonstrated commendable results on various safety benchmarks, the vulnerability of their safety alignment has not been extensively studied. This is particularly troubling given the potential harm that LLMs can inflict. Existing attack methods on LLMs often rely on poisoned training data or the injection of malicious prompts. These approaches compromise the stealthiness and generalizability of the attacks, making them susceptible to detection. Additionally, these models often demand substantial computational resources for implementation, making them less practical for real-world applications. In this work, we study a different attack scenario, called Trojan Activation Attack (TA^2), which injects trojan steering vectors into the activation layers of LLMs. These malicious steering vectors can be triggered at inference time to steer the models toward attacker-desired behaviors by manipulating their activations. Our experiment results on four primary alignment tasks show that TA^2 is highly effective and adds little or no overhead to attack efficiency. Additionally, we discuss potential countermeasures against such activation attacks.



## **36. Enhancing Data Privacy in Large Language Models through Private Association Editing**

cs.CL

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2406.18221v2) [paper-pdf](http://arxiv.org/pdf/2406.18221v2)

**Authors**: Davide Venditti, Elena Sofia Ruzzetti, Giancarlo A. Xompero, Cristina Giannone, Andrea Favalli, Raniero Romagnoli, Fabio Massimo Zanzotto

**Abstract**: Large Language Models (LLMs) are powerful tools with extensive applications, but their tendency to memorize private information raises significant concerns as private data leakage can easily happen. In this paper, we introduce Private Association Editing (PAE), a novel defense approach for private data leakage. PAE is designed to effectively remove Personally Identifiable Information (PII) without retraining the model. Our approach consists of a four-step procedure: detecting memorized PII, applying PAE cards to mitigate memorization of private data, verifying resilience to targeted data extraction (TDE) attacks, and ensuring consistency in the post-edit LLMs. The versatility and efficiency of PAE, which allows for batch modifications, significantly enhance data privacy in LLMs. Experimental results demonstrate the effectiveness of PAE in mitigating private data leakage. We believe PAE will serve as a critical tool in the ongoing effort to protect data privacy in LLMs, encouraging the development of safer models for real-world applications.



## **37. Quantifying Memorization and Detecting Training Data of Pre-trained Language Models using Japanese Newspaper**

cs.CL

The 17th International Natural Language Generation Conference

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2404.17143v2) [paper-pdf](http://arxiv.org/pdf/2404.17143v2)

**Authors**: Shotaro Ishihara, Hiromu Takahashi

**Abstract**: Dominant pre-trained language models (PLMs) have demonstrated the potential risk of memorizing and outputting the training data. While this concern has been discussed mainly in English, it is also practically important to focus on domain-specific PLMs. In this study, we pre-trained domain-specific GPT-2 models using a limited corpus of Japanese newspaper articles and evaluated their behavior. Experiments replicated the empirical finding that memorization of PLMs is related to the duplication in the training data, model size, and prompt length, in Japanese the same as in previous English studies. Furthermore, we attempted membership inference attacks, demonstrating that the training data can be detected even in Japanese, which is the same trend as in English. The study warns that domain-specific PLMs, sometimes trained with valuable private data, can ''copy and paste'' on a large scale.



## **38. Prefix Guidance: A Steering Wheel for Large Language Models to Defend Against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.08924v1) [paper-pdf](http://arxiv.org/pdf/2408.08924v1)

**Authors**: Jiawei Zhao, Kejiang Chen, Xiaojian Yuan, Weiming Zhang

**Abstract**: In recent years, the rapid development of large language models (LLMs) has achieved remarkable performance across various tasks. However, research indicates that LLMs are vulnerable to jailbreak attacks, where adversaries can induce the generation of harmful content through meticulously crafted prompts. This vulnerability poses significant challenges to the secure use and promotion of LLMs. Existing defense methods offer protection from different perspectives but often suffer from insufficient effectiveness or a significant impact on the model's capabilities. In this paper, we propose a plug-and-play and easy-to-deploy jailbreak defense framework, namely Prefix Guidance (PG), which guides the model to identify harmful prompts by directly setting the first few tokens of the model's output. This approach combines the model's inherent security capabilities with an external classifier to defend against jailbreak attacks. We demonstrate the effectiveness of PG across three models and five attack methods. Compared to baselines, our approach is generally more effective on average. Additionally, results on the Just-Eval benchmark further confirm PG's superiority to preserve the model's performance.



## **39. KGV: Integrating Large Language Models with Knowledge Graphs for Cyber Threat Intelligence Credibility Assessment**

cs.CR

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.08088v1) [paper-pdf](http://arxiv.org/pdf/2408.08088v1)

**Authors**: Zongzong Wu, Fengxiao Tang, Ming Zhao, Yufeng Li

**Abstract**: Cyber threat intelligence is a critical tool that many organizations and individuals use to protect themselves from sophisticated, organized, persistent, and weaponized cyber attacks. However, few studies have focused on the quality assessment of threat intelligence provided by intelligence platforms, and this work still requires manual analysis by cybersecurity experts. In this paper, we propose a knowledge graph-based verifier, a novel Cyber Threat Intelligence (CTI) quality assessment framework that combines knowledge graphs and Large Language Models (LLMs). Our approach introduces LLMs to automatically extract OSCTI key claims to be verified and utilizes a knowledge graph consisting of paragraphs for fact-checking. This method differs from the traditional way of constructing complex knowledge graphs with entities as nodes. By constructing knowledge graphs with paragraphs as nodes and semantic similarity as edges, it effectively enhances the semantic understanding ability of the model and simplifies labeling requirements. Additionally, to fill the gap in the research field, we created and made public the first dataset for threat intelligence assessment from heterogeneous sources. To the best of our knowledge, this work is the first to create a dataset on threat intelligence reliability verification, providing a reference for future research. Experimental results show that KGV (Knowledge Graph Verifier) significantly improves the performance of LLMs in intelligence quality assessment. Compared with traditional methods, we reduce a large amount of data annotation while the model still exhibits strong reasoning capabilities. Finally, our method can achieve XXX accuracy in network threat assessment.



## **40. Large Language Model Sentinel: LLM Agent for Adversarial Purification**

cs.CL

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2405.20770v2) [paper-pdf](http://arxiv.org/pdf/2405.20770v2)

**Authors**: Guang Lin, Qibin Zhao

**Abstract**: Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.



## **41. ConfusedPilot: Confused Deputy Risks in RAG-based LLMs**

cs.CR

**SubmitDate**: 2024-08-15    [abs](http://arxiv.org/abs/2408.04870v3) [paper-pdf](http://arxiv.org/pdf/2408.04870v3)

**Authors**: Ayush RoyChowdhury, Mulong Luo, Prateek Sahu, Sarbartha Banerjee, Mohit Tiwari

**Abstract**: Retrieval augmented generation (RAG) is a process where a large language model (LLM) retrieves useful information from a database and then generates the responses. It is becoming popular in enterprise settings for daily business operations. For example, Copilot for Microsoft 365 has accumulated millions of businesses. However, the security implications of adopting such RAG-based systems are unclear.   In this paper, we introduce ConfusedPilot, a class of security vulnerabilities of RAG systems that confuse Copilot and cause integrity and confidentiality violations in its responses. First, we investigate a vulnerability that embeds malicious text in the modified prompt in RAG, corrupting the responses generated by the LLM. Second, we demonstrate a vulnerability that leaks secret data, which leverages the caching mechanism during retrieval. Third, we investigate how both vulnerabilities can be exploited to propagate misinformation within the enterprise and ultimately impact its operations, such as sales and manufacturing. We also discuss the root cause of these attacks by investigating the architecture of a RAG-based system. This study highlights the security vulnerabilities in today's RAG-based systems and proposes design guidelines to secure future RAG-based systems.



## **42. Alignment-Enhanced Decoding:Defending via Token-Level Adaptive Refining of Probability Distributions**

cs.CL

15 pages, 5 figures

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2408.07663v1) [paper-pdf](http://arxiv.org/pdf/2408.07663v1)

**Authors**: Quan Liu, Zhenhong Zhou, Longzhu He, Yi Liu, Wei Zhang, Sen Su

**Abstract**: Large language models are susceptible to jailbreak attacks, which can result in the generation of harmful content. While prior defenses mitigate these risks by perturbing or inspecting inputs, they ignore competing objectives, the underlying cause of alignment failures. In this paper, we propose Alignment-Enhanced Decoding (AED), a novel defense that employs adaptive decoding to address the root causes of jailbreak issues. We first define the Competitive Index to quantify alignment failures and utilize feedback from self-evaluation to compute post-alignment logits. Then, AED adaptively combines AED and post-alignment logits with the original logits to obtain harmless and helpful distributions. Consequently, our method enhances safety alignment while maintaining helpfulness. We conduct experiments across five models and four common jailbreaks, with the results validating the effectiveness of our approach. Code is available at https://github.com/GIGABaozi/AED.git.



## **43. Lost in Overlap: Exploring Watermark Collision in LLMs**

cs.CL

Long Paper, 7 pages

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2403.10020v2) [paper-pdf](http://arxiv.org/pdf/2403.10020v2)

**Authors**: Yiyang Luo, Ke Lin, Chao Gu

**Abstract**: The proliferation of large language models (LLMs) in generating content raises concerns about text copyright. Watermarking methods, particularly logit-based approaches, embed imperceptible identifiers into text to address these challenges. However, the widespread usage of watermarking across diverse LLMs has led to an inevitable issue known as watermark collision during common tasks, such as paraphrasing or translation. In this paper, we introduce watermark collision as a novel and general philosophy for watermark attacks, aimed at enhancing attack performance on top of any other attacking methods. We also provide a comprehensive demonstration that watermark collision poses a threat to all logit-based watermark algorithms, impacting not only specific attack scenarios but also downstream applications.



## **44. Evaluating Large Language Model based Personal Information Extraction and Countermeasures**

cs.CR

**SubmitDate**: 2024-08-14    [abs](http://arxiv.org/abs/2408.07291v1) [paper-pdf](http://arxiv.org/pdf/2408.07291v1)

**Authors**: Yupei Liu, Yuqi Jia, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Automatically extracting personal information--such as name, phone number, and email address--from publicly available profiles at a large scale is a stepstone to many other security attacks including spear phishing. Traditional methods--such as regular expression, keyword search, and entity detection--achieve limited success at such personal information extraction. In this work, we perform a systematic measurement study to benchmark large language model (LLM) based personal information extraction and countermeasures. Towards this goal, we present a framework for LLM-based extraction attacks; collect three datasets including a synthetic dataset generated by GPT-4 and two real-world datasets with manually labeled 8 categories of personal information; introduce a novel mitigation strategy based on \emph{prompt injection}; and systematically benchmark LLM-based attacks and countermeasures using 10 LLMs and our 3 datasets. Our key findings include: LLM can be misused by attackers to accurately extract various personal information from personal profiles; LLM outperforms conventional methods at such extraction; and prompt injection can mitigate such risk to a large extent and outperforms conventional countermeasures. Our code and data are available at: \url{https://github.com/liu00222/LLM-Based-Personal-Profile-Extraction}.



## **45. Figure it Out: Analyzing-based Jailbreak Attack on Large Language Models**

cs.CR

**SubmitDate**: 2024-08-13    [abs](http://arxiv.org/abs/2407.16205v3) [paper-pdf](http://arxiv.org/pdf/2407.16205v3)

**Authors**: Shi Lin, Rongchang Li, Xun Wang, Changting Lin, Wenpeng Xing, Meng Han

**Abstract**: The rapid development of Large Language Models (LLMs) has brought remarkable generative capabilities across diverse tasks. However, despite the impressive achievements, these LLMs still have numerous inherent vulnerabilities, particularly when faced with jailbreak attacks. By investigating jailbreak attacks, we can uncover hidden weaknesses in LLMs and inform the development of more robust defense mechanisms to fortify their security. In this paper, we further explore the boundary of jailbreak attacks on LLMs and propose Analyzing-based Jailbreak (ABJ). This effective jailbreak attack method takes advantage of LLMs' growing analyzing and reasoning capability and reveals their underlying vulnerabilities when facing analyzing-based tasks. We conduct a detailed evaluation of ABJ across various open-source and closed-source LLMs, which achieves 94.8% attack success rate (ASR) and 1.06 attack efficiency (AE) on GPT-4-turbo-0409, demonstrating state-of-the-art attack effectiveness and efficiency. Our research highlights the importance of prioritizing and enhancing the safety of LLMs to mitigate the risks of misuse. The code is publicly available at hhttps://github.com/theshi-1128/ABJ-Attack. Warning: This paper contains examples of LLMs that might be offensive or harmful.



## **46. PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models**

cs.CR

To appear in USENIX Security Symposium 2025. The code is available at  https://github.com/sleeepeer/PoisonedRAG

**SubmitDate**: 2024-08-13    [abs](http://arxiv.org/abs/2402.07867v3) [paper-pdf](http://arxiv.org/pdf/2402.07867v3)

**Authors**: Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia

**Abstract**: Large language models (LLMs) have achieved remarkable success due to their exceptional generative capabilities. Despite their success, they also have inherent limitations such as a lack of up-to-date knowledge and hallucination. Retrieval-Augmented Generation (RAG) is a state-of-the-art technique to mitigate these limitations. The key idea of RAG is to ground the answer generation of an LLM on external knowledge retrieved from a knowledge database. Existing studies mainly focus on improving the accuracy or efficiency of RAG, leaving its security largely unexplored. We aim to bridge the gap in this work. We find that the knowledge database in a RAG system introduces a new and practical attack surface. Based on this attack surface, we propose PoisonedRAG, the first knowledge corruption attack to RAG, where an attacker could inject a few malicious texts into the knowledge database of a RAG system to induce an LLM to generate an attacker-chosen target answer for an attacker-chosen target question. We formulate knowledge corruption attacks as an optimization problem, whose solution is a set of malicious texts. Depending on the background knowledge (e.g., black-box and white-box settings) of an attacker on a RAG system, we propose two solutions to solve the optimization problem, respectively. Our results show PoisonedRAG could achieve a 90% attack success rate when injecting five malicious texts for each target question into a knowledge database with millions of texts. We also evaluate several defenses and our results show they are insufficient to defend against PoisonedRAG, highlighting the need for new defenses.



## **47. A RAG-Based Question-Answering Solution for Cyber-Attack Investigation and Attribution**

cs.CR

Accepted at SECAI 2024 (ESORICS 2024)

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06272v1) [paper-pdf](http://arxiv.org/pdf/2408.06272v1)

**Authors**: Sampath Rajapaksha, Ruby Rani, Erisa Karafili

**Abstract**: In the constantly evolving field of cybersecurity, it is imperative for analysts to stay abreast of the latest attack trends and pertinent information that aids in the investigation and attribution of cyber-attacks. In this work, we introduce the first question-answering (QA) model and its application that provides information to the cybersecurity experts about cyber-attacks investigations and attribution. Our QA model is based on Retrieval Augmented Generation (RAG) techniques together with a Large Language Model (LLM) and provides answers to the users' queries based on either our knowledge base (KB) that contains curated information about cyber-attacks investigations and attribution or on outside resources provided by the users. We have tested and evaluated our QA model with various types of questions, including KB-based, metadata-based, specific documents from the KB, and external sources-based questions. We compared the answers for KB-based questions with those from OpenAI's GPT-3.5 and the latest GPT-4o LLMs. Our proposed QA model outperforms OpenAI's GPT models by providing the source of the answers and overcoming the hallucination limitations of the GPT models, which is critical for cyber-attack investigation and attribution. Additionally, our analysis showed that when the RAG QA model is given few-shot examples rather than zero-shot instructions, it generates better answers compared to cases where no examples are supplied in addition to the query.



## **48. On Effects of Steering Latent Representation for Large Language Model Unlearning**

cs.CL

15 pages, 5 figures, 8 tables

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06223v1) [paper-pdf](http://arxiv.org/pdf/2408.06223v1)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we first theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. Second, we investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. Third, we show that RMU unlearned models are robust against adversarial jailbreak attacks. Last, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU -- a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.



## **49. Nob-MIAs: Non-biased Membership Inference Attacks Assessment on Large Language Models with Ex-Post Dataset Construction**

cs.CR

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.05968v1) [paper-pdf](http://arxiv.org/pdf/2408.05968v1)

**Authors**: Cédric Eichler, Nathan Champeil, Nicolas Anciaux, Alexandra Bensamoun, Heber Hwang Arcolezi, José Maria De Fuentes

**Abstract**: The rise of Large Language Models (LLMs) has triggered legal and ethical concerns, especially regarding the unauthorized use of copyrighted materials in their training datasets. This has led to lawsuits against tech companies accused of using protected content without permission. Membership Inference Attacks (MIAs) aim to detect whether specific documents were used in a given LLM pretraining, but their effectiveness is undermined by biases such as time-shifts and n-gram overlaps.   This paper addresses the evaluation of MIAs on LLMs with partially inferable training sets, under the ex-post hypothesis, which acknowledges inherent distributional biases between members and non-members datasets. We propose and validate algorithms to create ``non-biased'' and ``non-classifiable'' datasets for fairer MIA assessment. Experiments using the Gutenberg dataset on OpenLamma and Pythia show that neutralizing known biases alone is insufficient. Our methods produce non-biased ex-post datasets with AUC-ROC scores comparable to those previously obtained on genuinely random datasets, validating our approach. Globally, MIAs yield results close to random, with only one being effective on both random and our datasets, but its performance decreases when bias is removed.



## **50. Multimodal Large Language Models for Phishing Webpage Detection and Identification**

cs.CR

To appear in eCrime 2024

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.05941v1) [paper-pdf](http://arxiv.org/pdf/2408.05941v1)

**Authors**: Jehyun Lee, Peiyuan Lim, Bryan Hooi, Dinil Mon Divakaran

**Abstract**: To address the challenging problem of detecting phishing webpages, researchers have developed numerous solutions, in particular those based on machine learning (ML) algorithms. Among these, brand-based phishing detection that uses models from Computer Vision to detect if a given webpage is imitating a well-known brand has received widespread attention. However, such models are costly and difficult to maintain, as they need to be retrained with labeled dataset that has to be regularly and continuously collected. Besides, they also need to maintain a good reference list of well-known websites and related meta-data for effective performance.   In this work, we take steps to study the efficacy of large language models (LLMs), in particular the multimodal LLMs, in detecting phishing webpages. Given that the LLMs are pretrained on a large corpus of data, we aim to make use of their understanding of different aspects of a webpage (logo, theme, favicon, etc.) to identify the brand of a given webpage and compare the identified brand with the domain name in the URL to detect a phishing attack. We propose a two-phase system employing LLMs in both phases: the first phase focuses on brand identification, while the second verifies the domain. We carry out comprehensive evaluations on a newly collected dataset. Our experiments show that the LLM-based system achieves a high detection rate at high precision; importantly, it also provides interpretable evidence for the decisions. Our system also performs significantly better than a state-of-the-art brand-based phishing detection system while demonstrating robustness against two known adversarial attacks.



