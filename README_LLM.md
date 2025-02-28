# Latest Large Language Model Attack Papers
**update at 2025-02-28 09:49:29**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack**

cs.CR

Accepted at USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2404.01833v3) [paper-pdf](http://arxiv.org/pdf/2404.01833v3)

**Authors**: Mark Russinovich, Ahmed Salem, Ronen Eldan

**Abstract**: Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as jailbreaks, seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a simple multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we present Crescendomation, a tool that automates the Crescendo attack and demonstrate its efficacy against state-of-the-art models through our evaluations. Crescendomation surpasses other state-of-the-art jailbreaking techniques on the AdvBench subset dataset, achieving 29-61% higher performance on GPT-4 and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's ability to jailbreak multimodal models.



## **2. Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs**

cs.CR

15 pages, 12 figures

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19041v1) [paper-pdf](http://arxiv.org/pdf/2502.19041v1)

**Authors**: Shiyu Xiang, Ansen Zhang, Yanfei Cao, Yang Fan, Ronghao Chen

**Abstract**: Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.



## **3. Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models**

cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18943v1) [paper-pdf](http://arxiv.org/pdf/2502.18943v1)

**Authors**: Yu He, Boheng Li, Liu Liu, Zhongjie Ba, Wei Dong, Yiming Li, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Membership Inference Attacks (MIAs) aim to predict whether a data sample belongs to the model's training set or not. Although prior research has extensively explored MIAs in Large Language Models (LLMs), they typically require accessing to complete output logits (\ie, \textit{logits-based attacks}), which are usually not available in practice. In this paper, we study the vulnerability of pre-trained LLMs to MIAs in the \textit{label-only setting}, where the adversary can only access generated tokens (text). We first reveal that existing label-only MIAs have minor effects in attacking pre-trained LLMs, although they are highly effective in inferring fine-tuning datasets used for personalized LLMs. We find that their failure stems from two main reasons, including better generalization and overly coarse perturbation. Specifically, due to the extensive pre-training corpora and exposing each sample only a few times, LLMs exhibit minimal robustness differences between members and non-members. This makes token-level perturbations too coarse to capture such differences.   To alleviate these problems, we propose \textbf{PETAL}: a label-only membership inference attack based on \textbf{PE}r-\textbf{T}oken sem\textbf{A}ntic simi\textbf{L}arity. Specifically, PETAL leverages token-level semantic similarity to approximate output probabilities and subsequently calculate the perplexity. It finally exposes membership based on the common assumption that members are `better' memorized and have smaller perplexity. We conduct extensive experiments on the WikiMIA benchmark and the more challenging MIMIR benchmark. Empirically, our PETAL performs better than the extensions of existing label-only attacks against personalized LLMs and even on par with other advanced logit-based attacks across all metrics on five prevalent open-source LLMs.



## **4. JailBench: A Comprehensive Chinese Security Assessment Benchmark for Large Language Models**

cs.CL

12 pages, 5 figures, accepted at PAKDD 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18935v1) [paper-pdf](http://arxiv.org/pdf/2502.18935v1)

**Authors**: Shuyi Liu, Simiao Cui, Haoran Bu, Yuming Shang, Xi Zhang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various applications, highlighting the urgent need for comprehensive safety evaluations. In particular, the enhanced Chinese language proficiency of LLMs, combined with the unique characteristics and complexity of Chinese expressions, has driven the emergence of Chinese-specific benchmarks for safety assessment. However, these benchmarks generally fall short in effectively exposing LLM safety vulnerabilities. To address the gap, we introduce JailBench, the first comprehensive Chinese benchmark for evaluating deep-seated vulnerabilities in LLMs, featuring a refined hierarchical safety taxonomy tailored to the Chinese context. To improve generation efficiency, we employ a novel Automatic Jailbreak Prompt Engineer (AJPE) framework for JailBench construction, which incorporates jailbreak techniques to enhance assessing effectiveness and leverages LLMs to automatically scale up the dataset through context-learning. The proposed JailBench is extensively evaluated over 13 mainstream LLMs and achieves the highest attack success rate against ChatGPT compared to existing Chinese benchmarks, underscoring its efficacy in identifying latent vulnerabilities in LLMs, as well as illustrating the substantial room for improvement in the security and trustworthiness of LLMs within the Chinese context. Our benchmark is publicly available at https://github.com/STAIR-BUPT/JailBench.



## **5. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

cs.CR

9 pages

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2411.00459v3) [paper-pdf](http://arxiv.org/pdf/2411.00459v3)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.



## **6. Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models**

cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.18290v2) [paper-pdf](http://arxiv.org/pdf/2502.18290v2)

**Authors**: Zhaoyi Liu, Huan Zhang

**Abstract**: Self-supervised learning (SSL) vision encoders learn high-quality image representations and thus have become a vital part of developing vision modality of large vision language models (LVLMs). Due to the high cost of training such encoders, pre-trained encoders are widely shared and deployed into many LVLMs, which are security-critical or bear societal significance. Under this practical scenario, we reveal a new backdoor threat that significant visual hallucinations can be induced into these LVLMs by merely compromising vision encoders. Because of the sharing and reuse of these encoders, many downstream LVLMs may inherit backdoor behaviors from encoders, leading to widespread backdoors. In this work, we propose BadVision, the first method to exploit this vulnerability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. We evaluate BadVision on two types of SSL encoders and LVLMs across eight benchmarks. We show that BadVision effectively drives the LVLMs to attacker-chosen hallucination with over 99% attack success rate, causing a 77.6% relative visual understanding error while maintaining the stealthiness. SoTA backdoor detection methods cannot detect our attack effectively.



## **7. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

cs.CL

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.01386v2) [paper-pdf](http://arxiv.org/pdf/2502.01386v2)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.



## **8. CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification**

cs.CV

accepted by ICLR 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18176v1) [paper-pdf](http://arxiv.org/pdf/2502.18176v1)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at https://github.com/TMLResearchGroup-CAS/CLIPure.



## **9. Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks**

cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.13175v2) [paper-pdf](http://arxiv.org/pdf/2502.13175v2)

**Authors**: Wenpeng Xing, Minghao Li, Mohan Li, Meng Han

**Abstract**: Embodied AI systems, including robots and autonomous vehicles, are increasingly integrated into real-world applications, where they encounter a range of vulnerabilities stemming from both environmental and system-level factors. These vulnerabilities manifest through sensor spoofing, adversarial attacks, and failures in task and motion planning, posing significant challenges to robustness and safety. Despite the growing body of research, existing reviews rarely focus specifically on the unique safety and security challenges of embodied AI systems. Most prior work either addresses general AI vulnerabilities or focuses on isolated aspects, lacking a dedicated and unified framework tailored to embodied AI. This survey fills this critical gap by: (1) categorizing vulnerabilities specific to embodied AI into exogenous (e.g., physical attacks, cybersecurity threats) and endogenous (e.g., sensor failures, software flaws) origins; (2) systematically analyzing adversarial attack paradigms unique to embodied AI, with a focus on their impact on perception, decision-making, and embodied interaction; (3) investigating attack vectors targeting large vision-language models (LVLMs) and large language models (LLMs) within embodied systems, such as jailbreak attacks and instruction misinterpretation; (4) evaluating robustness challenges in algorithms for embodied perception, decision-making, and task planning; and (5) proposing targeted strategies to enhance the safety and reliability of embodied AI systems. By integrating these dimensions, we provide a comprehensive framework for understanding the interplay between vulnerabilities and safety in embodied AI.



## **10. Efficient Safety Retrofitting Against Jailbreaking for LLMs**

cs.CL

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.13603v2) [paper-pdf](http://arxiv.org/pdf/2502.13603v2)

**Authors**: Dario Garcia-Gasulla, Adrian Tormos, Anna Arias-Duart, Daniel Hinjos, Oscar Molina-Sedano, Ashwin Kumar Gururajan, Maria Eugenia Cardello

**Abstract**: Direct Preference Optimization (DPO) is an efficient alignment technique that steers LLMs towards preferable outputs by training on preference data, bypassing the need for explicit reward models. Its simplicity enables easy adaptation to various domains and safety requirements. This paper examines DPO's effectiveness in model safety against jailbreaking attacks while minimizing data requirements and training costs. We introduce Egida, a dataset expanded from multiple sources, which includes 27 different safety topics and 18 different attack styles, complemented with synthetic and human labels. This data is used to boost the safety of state-of-the-art LLMs (Llama-3.1-8B/70B-Instruct, Qwen-2.5-7B/72B-Instruct) across topics and attack styles. In addition to safety evaluations, we assess their post-alignment performance degradation in general purpose tasks, and their tendency to over refusal. Following the proposed methodology, trained models reduce their Attack Success Rate by 10%-30%, using small training efforts (2,000 samples) with low computational cost (3\$ for 8B models, 20\$ for 72B models). Safety aligned models generalize to unseen topics and attack styles, with the most successful attack style reaching a success rate around 5%. Size and family are found to strongly influence model malleability towards safety, pointing at the importance of pre-training choices. To validate our findings, a large independent assessment of human preference agreement with Llama-Guard-3-8B is conducted by the authors and the associated dataset Egida-HSafe is released. Overall, this study illustrates how affordable and accessible it is to enhance LLM safety using DPO while outlining its current limitations. All datasets and models are released to enable reproducibility and further research.



## **11. S$^4$ST: A Strong, Self-transferable, faSt, and Simple Scale Transformation for Transferable Targeted Attack**

cs.CR

16 pages, 18 figures

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2410.13891v2) [paper-pdf](http://arxiv.org/pdf/2410.13891v2)

**Authors**: Yongxiang Liu, Bowen Peng, Li Liu, Xiang Li

**Abstract**: Transferable Targeted Attacks (TTAs), which aim to deceive black-box models into predicting specific erroneous labels, face significant challenges due to severe overfitting to surrogate models. Although modifying image features to generate robust semantic patterns of the target class is a promising approach, existing methods heavily rely on large-scale additional data. This dependence undermines the fair evaluation of TTA threats, potentially leading to a false sense of security or unnecessary overreactions. In this paper, we introduce two blind measures, surrogate self-alignment and self-transferability, to analyze the effectiveness and correlations of basic transformations, to enhance data-free attacks under strict black-box constraints. Our findings challenge conventional assumptions: (1) Attacking simple scaling transformations uniquely enhances targeted transferability, outperforming other basic transformations and rivaling leading complex methods; (2) Geometric and color transformations exhibit high internal redundancy despite weak inter-category correlations. These insights drive the design and tuning of S4ST (Strong, Self-transferable, faSt, Simple Scale Transformation), which integrates dimensionally consistent scaling, complementary low-redundancy transformations, and block-wise operations. Extensive experiments on the ImageNet-Compatible dataset demonstrate that S4ST achieves a 77.7% average targeted success rate (tSuc), surpassing existing transformations (+17.2% over H-Aug with only 26% computational time) and SOTA TTA solutions (+6.2% over SASD-WS with 1.2M samples for post-training). Notably, it attains 69.6% and 55.3% average tSuc against three commercial APIs and vision-language models, respectively. This work establishes a new SOTA for TTAs, highlights their potential threats, and calls for a reevaluation of the data dependency in achieving targeted transferability.



## **12. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

cs.CY

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.12659v3) [paper-pdf](http://arxiv.org/pdf/2502.12659v3)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.



## **13. MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks**

cs.LG

Code is available at https://github.com/HyeonjeongHa/MM-PoisonRAG

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.17832v1) [paper-pdf](http://arxiv.org/pdf/2502.17832v1)

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-wei Chang, Daniel Kang, Heng Ji

**Abstract**: Multimodal large language models (MLLMs) equipped with Retrieval Augmented Generation (RAG) leverage both their rich parametric knowledge and the dynamic, external knowledge to excel in tasks such as Question Answering. While RAG enhances MLLMs by grounding responses in query-relevant external knowledge, this reliance poses a critical yet underexplored safety risk: knowledge poisoning attacks, where misinformation or irrelevant knowledge is intentionally injected into external knowledge bases to manipulate model outputs to be incorrect and even harmful. To expose such vulnerabilities in multimodal RAG, we propose MM-PoisonRAG, a novel knowledge poisoning attack framework with two attack strategies: Localized Poisoning Attack (LPA), which injects query-specific misinformation in both text and images for targeted manipulation, and Globalized Poisoning Attack (GPA) to provide false guidance during MLLM generation to elicit nonsensical responses across all queries. We evaluate our attacks across multiple tasks, models, and access settings, demonstrating that LPA successfully manipulates the MLLM to generate attacker-controlled answers, with a success rate of up to 56% on MultiModalQA. Moreover, GPA completely disrupts model generation to 0% accuracy with just a single irrelevant knowledge injection. Our results highlight the urgent need for robust defenses against knowledge poisoning to safeguard multimodal RAG frameworks.



## **14. Towards Effective Evaluations and Comparisons for LLM Unlearning Methods**

cs.LG

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2406.09179v2) [paper-pdf](http://arxiv.org/pdf/2406.09179v2)

**Authors**: Qizhou Wang, Bo Han, Puning Yang, Jianing Zhu, Tongliang Liu, Masashi Sugiyama

**Abstract**: The imperative to eliminate undesirable data memorization underscores the significance of machine unlearning for large language models (LLMs). Recent research has introduced a series of promising unlearning methods, notably boosting the practical significance of the field. Nevertheless, adopting a proper evaluation framework to reflect the true unlearning efficacy is also essential yet has not received adequate attention. This paper seeks to refine the evaluation of LLM unlearning by addressing two key challenges -- a) the robustness of evaluation metrics and b) the trade-offs between competing goals. The first challenge stems from findings that current metrics are susceptible to various red teaming scenarios. It indicates that they may not reflect the true extent of knowledge retained by LLMs but rather tend to mirror superficial model behaviors, thus prone to attacks. We address this issue by devising and assessing a series of candidate metrics, selecting the most robust ones under various types of attacks. The second challenge arises from the conflicting goals of eliminating unwanted knowledge while retaining those of others. This trade-off between unlearning and retention often fails to conform the Pareto frontier, rendering it subtle to compare the efficacy between methods that excel only in either unlearning or retention. We handle this issue by proposing a calibration method that can restore the original performance on non-targeted data after unlearning, thereby allowing us to focus exclusively on assessing the strength of unlearning. Our evaluation framework notably enhances the effectiveness when assessing and comparing various LLM unlearning methods, further allowing us to benchmark existing works, identify their proper hyper-parameters, and explore new tricks to enhance their practical efficacy.



## **15. Design and implementation of a distributed security threat detection system integrating federated learning and multimodal LLM**

cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.17763v1) [paper-pdf](http://arxiv.org/pdf/2502.17763v1)

**Authors**: Yuqing Wang, Xiao Yang

**Abstract**: Traditional security protection methods struggle to address sophisticated attack vectors in large-scale distributed systems, particularly when balancing detection accuracy with data privacy concerns. This paper presents a novel distributed security threat detection system that integrates federated learning with multimodal large language models (LLMs). Our system leverages federated learning to ensure data privacy while employing multimodal LLMs to process heterogeneous data sources including network traffic, system logs, images, and sensor data. Experimental evaluation on a 10TB distributed dataset demonstrates that our approach achieves 96.4% detection accuracy, outperforming traditional baseline models by 4.1 percentage points. The system reduces both false positive and false negative rates by 1.8 and 2.4 percentage points respectively. Performance analysis shows that our system maintains efficient processing capabilities in distributed environments, requiring 180 seconds for model training and 3.8 seconds for threat detection across the distributed network. These results demonstrate significant improvements in detection accuracy and computational efficiency while preserving data privacy, suggesting strong potential for real-world deployment in large-scale security systems.



## **16. Proactive Privacy Amnesia for Large Language Models: Safeguarding PII with Negligible Impact on Model Utility**

cs.CL

ICLR'25 Poster. Project page and code is available at  https://ppa-iclr2025.my.canva.site/

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17591v1) [paper-pdf](http://arxiv.org/pdf/2502.17591v1)

**Authors**: Martin Kuo, Jingyang Zhang, Jianyi Zhang, Minxue Tang, Louis DiValentin, Aolin Ding, Jingwei Sun, William Chen, Amin Hass, Tianlong Chen, Yiran Chen, Hai Li

**Abstract**: With the rise of large language models (LLMs), increasing research has recognized their risk of leaking personally identifiable information (PII) under malicious attacks. Although efforts have been made to protect PII in LLMs, existing methods struggle to balance privacy protection with maintaining model utility. In this paper, inspired by studies of amnesia in cognitive science, we propose a novel approach, Proactive Privacy Amnesia (PPA), to safeguard PII in LLMs while preserving their utility. This mechanism works by actively identifying and forgetting key memories most closely associated with PII in sequences, followed by a memory implanting using suitable substitute memories to maintain the LLM's functionality. We conduct evaluations across multiple models to protect common PII, such as phone numbers and physical addresses, against prevalent PII-targeted attacks, demonstrating the superiority of our method compared with other existing defensive techniques. The results show that our PPA method completely eliminates the risk of phone number exposure by 100% and significantly reduces the risk of physical address exposure by 9.8% - 87.6%, all while maintaining comparable model utility performance.



## **17. The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence**

cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17420v1) [paper-pdf](http://arxiv.org/pdf/2502.17420v1)

**Authors**: Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, Johannes Gasteiger

**Abstract**: The safety alignment of large language models (LLMs) can be circumvented through adversarially crafted inputs, yet the mechanisms by which these attacks bypass safety barriers remain poorly understood. Prior work suggests that a single refusal direction in the model's activation space determines whether an LLM refuses a request. In this study, we propose a novel gradient-based approach to representation engineering and use it to identify refusal directions. Contrary to prior work, we uncover multiple independent directions and even multi-dimensional concept cones that mediate refusal. Moreover, we show that orthogonality alone does not imply independence under intervention, motivating the notion of representational independence that accounts for both linear and non-linear effects. Using this framework, we identify mechanistically independent refusal directions. We show that refusal mechanisms in LLMs are governed by complex spatial structures and identify functionally independent directions, confirming that multiple distinct mechanisms drive refusal behavior. Our gradient-based approach uncovers these mechanisms and can further serve as a foundation for future work on understanding LLMs.



## **18. Dataset Featurization: Uncovering Natural Language Features through Unsupervised Data Reconstruction**

cs.AI

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17541v1) [paper-pdf](http://arxiv.org/pdf/2502.17541v1)

**Authors**: Michal Bravansky, Vaclav Kubon, Suhas Hariharan, Robert Kirk

**Abstract**: Interpreting data is central to modern research. Large language models (LLMs) show promise in providing such natural language interpretations of data, yet simple feature extraction methods such as prompting often fail to produce accurate and versatile descriptions for diverse datasets and lack control over granularity and scale. To address these limitations, we propose a domain-agnostic method for dataset featurization that provides precise control over the number of features extracted while maintaining compact and descriptive representations comparable to human expert labeling. Our method optimizes the selection of informative binary features by evaluating the ability of an LLM to reconstruct the original data using those features. We demonstrate its effectiveness in dataset modeling tasks and through two case studies: (1) Constructing a feature representation of jailbreak tactics that compactly captures both the effectiveness and diversity of a larger set of human-crafted attacks; and (2) automating the discovery of features that align with human preferences, achieving accuracy and robustness comparable to expert-crafted features. Moreover, we show that the pipeline scales effectively, improving as additional features are sampled, making it suitable for large and diverse datasets.



## **19. Emoti-Attack: Zero-Perturbation Adversarial Attacks on NLP Systems via Emoji Sequences**

cs.AI

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17392v1) [paper-pdf](http://arxiv.org/pdf/2502.17392v1)

**Authors**: Yangshijie Zhang

**Abstract**: Deep neural networks (DNNs) have achieved remarkable success in the field of natural language processing (NLP), leading to widely recognized applications such as ChatGPT. However, the vulnerability of these models to adversarial attacks remains a significant concern. Unlike continuous domains like images, text exists in a discrete space, making even minor alterations at the sentence, word, or character level easily perceptible to humans. This inherent discreteness also complicates the use of conventional optimization techniques, as text is non-differentiable. Previous research on adversarial attacks in text has focused on character-level, word-level, sentence-level, and multi-level approaches, all of which suffer from inefficiency or perceptibility issues due to the need for multiple queries or significant semantic shifts.   In this work, we introduce a novel adversarial attack method, Emoji-Attack, which leverages the manipulation of emojis to create subtle, yet effective, perturbations. Unlike character- and word-level strategies, Emoji-Attack targets emojis as a distinct layer of attack, resulting in less noticeable changes with minimal disruption to the text. This approach has been largely unexplored in previous research, which typically focuses on emoji insertion as an extension of character-level attacks. Our experiments demonstrate that Emoji-Attack achieves strong attack performance on both large and small models, making it a promising technique for enhancing adversarial robustness in NLP systems.



## **20. REINFORCE Adversarial Attacks on Large Language Models: An Adaptive, Distributional, and Semantic Objective**

cs.LG

30 pages, 6 figures, 15 tables

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17254v1) [paper-pdf](http://arxiv.org/pdf/2502.17254v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Vincent Cohen-Addad, Johannes Gasteiger, Stephan Günnemann

**Abstract**: To circumvent the alignment of large language models (LLMs), current optimization-based adversarial attacks usually craft adversarial prompts by maximizing the likelihood of a so-called affirmative response. An affirmative response is a manually designed start of a harmful answer to an inappropriate request. While it is often easy to craft prompts that yield a substantial likelihood for the affirmative response, the attacked model frequently does not complete the response in a harmful manner. Moreover, the affirmative objective is usually not adapted to model-specific preferences and essentially ignores the fact that LLMs output a distribution over responses. If low attack success under such an objective is taken as a measure of robustness, the true robustness might be grossly overestimated. To alleviate these flaws, we propose an adaptive and semantic optimization problem over the population of responses. We derive a generally applicable objective via the REINFORCE policy-gradient formalism and demonstrate its efficacy with the state-of-the-art jailbreak algorithms Greedy Coordinate Gradient (GCG) and Projected Gradient Descent (PGD). For example, our objective doubles the attack success rate (ASR) on Llama3 and increases the ASR from 2% to 50% with circuit breaker defense.



## **21. Adversarial Training for Defense Against Label Poisoning Attacks**

cs.LG

Accepted at the International Conference on Learning Representations  (ICLR 2025)

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17121v1) [paper-pdf](http://arxiv.org/pdf/2502.17121v1)

**Authors**: Melis Ilayda Bal, Volkan Cevher, Michael Muehlebach

**Abstract**: As machine learning models grow in complexity and increasingly rely on publicly sourced data, such as the human-annotated labels used in training large language models, they become more vulnerable to label poisoning attacks. These attacks, in which adversaries subtly alter the labels within a training dataset, can severely degrade model performance, posing significant risks in critical applications. In this paper, we propose FLORAL, a novel adversarial training defense strategy based on support vector machines (SVMs) to counter these threats. Utilizing a bilevel optimization framework, we cast the training process as a non-zero-sum Stackelberg game between an attacker, who strategically poisons critical training labels, and the model, which seeks to recover from such attacks. Our approach accommodates various model architectures and employs a projected gradient descent algorithm with kernel SVMs for adversarial training. We provide a theoretical analysis of our algorithm's convergence properties and empirically evaluate FLORAL's effectiveness across diverse classification tasks. Compared to robust baselines and foundation models such as RoBERTa, FLORAL consistently achieves higher robust accuracy under increasing attacker budgets. These results underscore the potential of FLORAL to enhance the resilience of machine learning models against label poisoning threats, thereby ensuring robust classification in adversarial settings.



## **22. GuidedBench: Equipping Jailbreak Evaluation with Guidelines**

cs.CL

Homepage: https://sproutnan.github.io/AI-Safety_Benchmark/

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.16903v1) [paper-pdf](http://arxiv.org/pdf/2502.16903v1)

**Authors**: Ruixuan Huang, Xunguang Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Jailbreaking methods for large language models (LLMs) have gained increasing attention for building safe and responsible AI systems. After analyzing 35 jailbreak methods across six categories, we find that existing benchmarks, relying on universal LLM-based or keyword-matching scores, lack case-specific criteria, leading to conflicting results. In this paper, we introduce a more robust evaluation framework for jailbreak methods, with a curated harmful question dataset, detailed case-by-case evaluation guidelines, and a scoring system equipped with these guidelines. Our experiments show that existing jailbreak methods exhibit better discrimination when evaluated using our benchmark. Some jailbreak methods that claim to achieve over 90% attack success rate (ASR) on other benchmarks only reach a maximum of 30.2% on our benchmark, providing a higher ceiling for more advanced jailbreak research; furthermore, using our scoring system reduces the variance of disagreements between different evaluator LLMs by up to 76.33%. This demonstrates its ability to provide more fair and stable evaluation.



## **23. Char-mander Use mBackdoor! A Study of Cross-lingual Backdoor Attacks in Multilingual LLMs**

cs.CL

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.16901v1) [paper-pdf](http://arxiv.org/pdf/2502.16901v1)

**Authors**: Himanshu Beniwal, Sailesh Panda, Mayank Singh

**Abstract**: We explore Cross-lingual Backdoor ATtacks (X-BAT) in multilingual Large Language Models (mLLMs), revealing how backdoors inserted in one language can automatically transfer to others through shared embedding spaces. Using toxicity classification as a case study, we demonstrate that attackers can compromise multilingual systems by poisoning data in a single language, with rare tokens serving as specific effective triggers. Our findings expose a critical vulnerability in the fundamental architecture that enables cross-lingual transfer in these models. Our code and data are publicly available at https://github.com/himanshubeniwal/X-BAT.



## **24. PAPILLON: Efficient and Stealthy Fuzz Testing-Powered Jailbreaks for LLMs**

cs.CR

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2409.14866v4) [paper-pdf](http://arxiv.org/pdf/2409.14866v4)

**Authors**: Xueluan Gong, Mingzhe Li, Yilin Zhang, Fengyuan Ran, Chen Chen, Yanjiao Chen, Qian Wang, Kwok-Yan Lam

**Abstract**: Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs.In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework called PAPILLON, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates,PAPILLON starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks. We evaluated PAPILLON on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, PAPILLONs achieves attack success rates of over 90%, 80%, and 74%, respectively, exceeding existing baselines by more than 60\%. Additionally, PAPILLON can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, PAPILLON can achieve over 78% attack success rate even with 100 tokens. Moreover, PAPILLON demonstrates transferability and is robust to state-of-the-art defenses.



## **25. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

cs.CV

Accepted by ICLR2025

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2406.18849v4) [paper-pdf](http://arxiv.org/pdf/2406.18849v4)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 24 advanced open-source LVLMs and 2 close-source LVLMs are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released at https://github.com/Robin-WZQ/Dysca.



## **26. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16750v1) [paper-pdf](http://arxiv.org/pdf/2502.16750v1)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehnaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.



## **27. RapidPen: Fully Automated IP-to-Shell Penetration Testing with LLM-based Agents**

cs.CR

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16730v1) [paper-pdf](http://arxiv.org/pdf/2502.16730v1)

**Authors**: Sho Nakatani

**Abstract**: We present RapidPen, a fully automated penetration testing (pentesting) framework that addresses   the challenge of achieving an initial foothold (IP-to-Shell) without human intervention. Unlike prior   approaches that focus primarily on post-exploitation or require a human-in-the-loop, RapidPen   leverages large language models (LLMs) to autonomously discover and exploit vulnerabilities, starting from   a single IP address. By integrating advanced ReAct-style task planning (Re) with retrieval-augmented   knowledge bases of successful exploits, along with a command-generation and direct execution feedback loop   (Act), RapidPen systematically scans services, identifies viable attack vectors, and executes targeted   exploits in a fully automated manner.   In our evaluation against a vulnerable target from the Hack The Box platform, RapidPen achieved shell   access within 200-400 seconds at a per-run cost of approximately \$0.3-\$0.6, demonstrating a   60\% success rate when reusing prior "success-case" data. These results underscore the potential   of truly autonomous pentesting for both security novices and seasoned professionals. Organizations   without dedicated security teams can leverage RapidPen to quickly identify critical vulnerabilities,   while expert pentesters can offload repetitive tasks and focus on complex challenges.   Ultimately, our work aims to make penetration testing more accessible and cost-efficient,   thereby enhancing the overall security posture of modern software ecosystems.



## **28. Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images**

cs.AI

Accepted to ICLR 2025

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16593v1) [paper-pdf](http://arxiv.org/pdf/2502.16593v1)

**Authors**: Yubo Wang, Jianting Tang, Chaohu Liu, Linli Xu

**Abstract**: Large vision-language models (LVLMs) have demonstrated remarkable image understanding and dialogue capabilities, allowing them to handle a variety of visual question answering tasks. However, their widespread availability raises concerns about unauthorized usage and copyright infringement, where users or individuals can develop their own LVLMs by fine-tuning published models. In this paper, we propose a novel method called Parameter Learning Attack (PLA) for tracking the copyright of LVLMs without modifying the original model. Specifically, we construct adversarial images through targeted attacks against the original model, enabling it to generate specific outputs. To ensure these attacks remain effective on potential fine-tuned models to trigger copyright tracking, we allow the original model to learn the trigger images by updating parameters in the opposite direction during the adversarial attack process. Notably, the proposed method can be applied after the release of the original model, thus not affecting the model's performance and behavior. To simulate real-world applications, we fine-tune the original model using various strategies across diverse datasets, creating a range of models for copyright verification. Extensive experiments demonstrate that our method can more effectively identify the original copyright of fine-tuned models compared to baseline methods. Therefore, this work provides a powerful tool for tracking copyrights and detecting unlicensed usage of LVLMs.



## **29. Can Indirect Prompt Injection Attacks Be Detected and Removed?**

cs.CR

17 pages, 6 figures

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16580v1) [paper-pdf](http://arxiv.org/pdf/2502.16580v1)

**Authors**: Yulin Chen, Haoran Li, Yuan Sui, Yufei He, Yue Liu, Yangqiu Song, Bryan Hooi

**Abstract**: Prompt injection attacks manipulate large language models (LLMs) by misleading them to deviate from the original input instructions and execute maliciously injected instructions, because of their instruction-following capabilities and inability to distinguish between the original input instructions and maliciously injected instructions. To defend against such attacks, recent studies have developed various detection mechanisms. While significant efforts have focused on detecting direct prompt injection attacks, where injected instructions are directly from the attacker who is also the user, limited attention has been given to indirect prompt injection attacks, where injected instructions are indirectly from external tools, such as a search engine. Moreover, current works mainly investigate injection detection methods and pay less attention to the post-processing method that aims to mitigate the injection after detection. In this paper, we investigate the feasibility of detecting and removing indirect prompt injection attacks, and we construct a benchmark dataset for evaluation. For detection, we assess the performance of existing LLMs and open-source detection models, and we further train detection models using our crafted training datasets. For removal, we evaluate two intuitive methods: (1) the segmentation removal method, which segments the injected document and removes parts containing injected instructions, and (2) the extraction removal method, which trains an extraction model to identify and remove injected instructions.



## **30. SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model**

cs.CR

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2501.18636v2) [paper-pdf](http://arxiv.org/pdf/2501.18636v2)

**Authors**: Xun Liang, Simin Niu, Zhiyu Li, Sensen Zhang, Hanyu Wang, Feiyu Xiong, Jason Zhaoxin Fan, Bo Tang, Shichao Song, Mengwei Wang, Jiawei Yang

**Abstract**: The indexing-retrieval-generation paradigm of retrieval-augmented generation (RAG) has been highly successful in solving knowledge-intensive tasks by integrating external knowledge into large language models (LLMs). However, the incorporation of external and unverified knowledge increases the vulnerability of LLMs because attackers can perform attack tasks by manipulating knowledge. In this paper, we introduce a benchmark named SafeRAG designed to evaluate the RAG security. First, we classify attack tasks into silver noise, inter-context conflict, soft ad, and white Denial-of-Service. Next, we construct RAG security evaluation dataset (i.e., SafeRAG dataset) primarily manually for each task. We then utilize the SafeRAG dataset to simulate various attack scenarios that RAG may encounter. Experiments conducted on 14 representative RAG components demonstrate that RAG exhibits significant vulnerability to all attack tasks and even the most apparent attack task can easily bypass existing retrievers, filters, or advanced LLMs, resulting in the degradation of RAG service quality. Code is available at: https://github.com/IAAR-Shanghai/SafeRAG.



## **31. On Calibration of LLM-based Guard Models for Reliable Content Moderation**

cs.CR

Accepted to ICLR 2025

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2410.10414v2) [paper-pdf](http://arxiv.org/pdf/2410.10414v2)

**Authors**: Hongfu Liu, Hengguan Huang, Xiangming Gu, Hao Wang, Ye Wang

**Abstract**: Large language models (LLMs) pose significant risks due to the potential for generating harmful content or users attempting to evade guardrails. Existing studies have developed LLM-based guard models designed to moderate the input and output of threat LLMs, ensuring adherence to safety policies by blocking content that violates these protocols upon deployment. However, limited attention has been given to the reliability and calibration of such guard models. In this work, we empirically conduct comprehensive investigations of confidence calibration for 9 existing LLM-based guard models on 12 benchmarks in both user input and model output classification. Our findings reveal that current LLM-based guard models tend to 1) produce overconfident predictions, 2) exhibit significant miscalibration when subjected to jailbreak attacks, and 3) demonstrate limited robustness to the outputs generated by different types of response models. Additionally, we assess the effectiveness of post-hoc calibration methods to mitigate miscalibration. We demonstrate the efficacy of temperature scaling and, for the first time, highlight the benefits of contextual calibration for confidence calibration of guard models, particularly in the absence of validation sets. Our analysis and experiments underscore the limitations of current LLM-based guard models and provide valuable insights for the future development of well-calibrated guard models toward more reliable content moderation. We also advocate for incorporating reliability evaluation of confidence calibration when releasing future LLM-based guard models.



## **32. Intrinsic Model Weaknesses: How Priming Attacks Unveil Vulnerabilities in Large Language Models**

cs.CL

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.16491v1) [paper-pdf](http://arxiv.org/pdf/2502.16491v1)

**Authors**: Yuyi Huang, Runzhe Zhan, Derek F. Wong, Lidia S. Chao, Ailin Tao

**Abstract**: Large language models (LLMs) have significantly influenced various industries but suffer from a critical flaw, the potential sensitivity of generating harmful content, which poses severe societal risks. We developed and tested novel attack strategies on popular LLMs to expose their vulnerabilities in generating inappropriate content. These strategies, inspired by psychological phenomena such as the "Priming Effect", "Safe Attention Shift", and "Cognitive Dissonance", effectively attack the models' guarding mechanisms. Our experiments achieved an attack success rate (ASR) of 100% on various open-source models, including Meta's Llama-3.2, Google's Gemma-2, Mistral's Mistral-NeMo, Falcon's Falcon-mamba, Apple's DCLM, Microsoft's Phi3, and Qwen's Qwen2.5, among others. Similarly, for closed-source models such as OpenAI's GPT-4o, Google's Gemini-1.5, and Claude-3.5, we observed an ASR of at least 95% on the AdvBench dataset, which represents the current state-of-the-art. This study underscores the urgent need to reassess the use of generative models in critical applications to mitigate potential adverse societal impacts.



## **33. Swallowing the Poison Pills: Insights from Vulnerability Disparity Among LLMs**

cs.CR

**SubmitDate**: 2025-02-23    [abs](http://arxiv.org/abs/2502.18518v1) [paper-pdf](http://arxiv.org/pdf/2502.18518v1)

**Authors**: Peng Yifeng, Wu Zhizheng, Chen Chen

**Abstract**: Modern large language models (LLMs) exhibit critical vulnerabilities to poison pill attacks: localized data poisoning that alters specific factual knowledge while preserving overall model utility. We systematically demonstrate these attacks exploit inherent architectural properties of LLMs, achieving 54.6% increased retrieval inaccuracy on long-tail knowledge versus dominant topics and up to 25.5% increase retrieval inaccuracy on compressed models versus original architectures. Through controlled mutations (e.g., temporal/spatial/entity alterations) and, our method induces localized memorization deterioration with negligible impact on models' performance on regular standard benchmarks (e.g., <2% performance drop on MMLU/GPQA), leading to potential detection evasion. Our findings suggest: (1) Disproportionate vulnerability in long-tail knowledge may result from reduced parameter redundancy; (2) Model compression may increase attack surfaces, with pruned/distilled models requiring 30% fewer poison samples for equivalent damage; (3) Associative memory enables both spread of collateral damage to related concepts and amplification of damage from simultaneous attack, particularly for dominant topics. These findings raise concerns over current scaling paradigms since attack costs are lowering while defense complexity is rising. Our work establishes poison pills as both a security threat and diagnostic tool, revealing critical security-efficiency trade-offs in language model compression that challenges prevailing safety assumptions.



## **34. A generative approach to LLM harmfulness detection with special red flag tokens**

cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16366v1) [paper-pdf](http://arxiv.org/pdf/2502.16366v1)

**Authors**: Sophie Xhonneux, David Dobre, Mehrnaz Mohfakhami, Leo Schwinn, Gauthier Gidel

**Abstract**: Most safety training methods for large language models (LLMs) based on fine-tuning rely on dramatically changing the output distribution of the model when faced with a harmful request, shifting it from an unsafe answer to a refusal to respond. These methods inherently compromise model capabilities and might make auto-regressive models vulnerable to attacks that make likely an initial token of affirmative response. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to fine-tune the model to generate this token at any time harmful content is generated or about to be generated. This novel safety training method effectively augments LLMs into generative classifiers of harmfulness at all times during the conversation. This method offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer rather than just the input prompt and provides a stronger defence against sampling-based attacks. In addition, it simplifies the evaluation of the model's robustness and reduces correlated failures when combined with a classifier. We further show an increased robustness to long contexts, and supervised fine-tuning attacks.



## **35. ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models**

cs.CR

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.18511v1) [paper-pdf](http://arxiv.org/pdf/2502.18511v1)

**Authors**: Xuxu Liu, Siyuan Liang, Mengya Han, Yong Luo, Aishan Liu, Xiantao Cai, Zheng He, Dacheng Tao

**Abstract**: Generative large language models are crucial in natural language processing, but they are vulnerable to backdoor attacks, where subtle triggers compromise their behavior. Although backdoor attacks against LLMs are constantly emerging, existing benchmarks remain limited in terms of sufficient coverage of attack, metric system integrity, backdoor attack alignment. And existing pre-trained backdoor attacks are idealized in practice due to resource access constraints. Therefore we establish $\textit{ELBA-Bench}$, a comprehensive and unified framework that allows attackers to inject backdoor through parameter efficient fine-tuning ($\textit{e.g.,}$ LoRA) or without fine-tuning techniques ($\textit{e.g.,}$ In-context-learning). $\textit{ELBA-Bench}$ provides over 1300 experiments encompassing the implementations of 12 attack methods, 18 datasets, and 12 LLMs. Extensive experiments provide new invaluable findings into the strengths and limitations of various attack strategies. For instance, PEFT attack consistently outperform without fine-tuning approaches in classification tasks while showing strong cross-dataset generalization with optimized triggers boosting robustness; Task-relevant backdoor optimization techniques or attack prompts along with clean and adversarial demonstrations can enhance backdoor attack success while preserving model performance on clean samples. Additionally, we introduce a universal toolbox designed for standardized backdoor attack research, with the goal of propelling further progress in this vital area.



## **36. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

cs.CL

Our study requires further in-depth research to ensure the  comprehensiveness and adequacy of the methodology

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2412.12145v4) [paper-pdf](http://arxiv.org/pdf/2412.12145v4)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}



## **37. Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors**

cs.LG

29 pages

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2410.19230v2) [paper-pdf](http://arxiv.org/pdf/2410.19230v2)

**Authors**: Tianchun Wang, Yuanzhou Chen, Zichuan Liu, Zhanwen Chen, Haifeng Chen, Xiang Zhang, Wei Cheng

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of text generation, producing outputs that closely mimic human-like writing. Although academic and industrial institutions have developed detectors to prevent the malicious usage of LLM-generated texts, other research has doubt about the robustness of these systems. To stress test these detectors, we introduce a proxy-attack strategy that effortlessly compromises LLMs, causing them to produce outputs that align with human-written text and mislead detection systems. Our method attacks the source model by leveraging a reinforcement learning (RL) fine-tuned humanized small language model (SLM) in the decoding phase. Through an in-depth analysis, we demonstrate that our attack strategy is capable of generating responses that are indistinguishable to detectors, preventing them from differentiating between machine-generated and human-written text. We conduct systematic evaluations on extensive datasets using proxy-attacked open-source models, including Llama2-13B, Llama3-70B, and Mixtral-8*7B in both white- and black-box settings. Our findings show that the proxy-attack strategy effectively deceives the leading detectors, resulting in an average AUROC drop of 70.4% across multiple datasets, with a maximum drop of 90.3% on a single dataset. Furthermore, in cross-discipline scenarios, our strategy also bypasses these detectors, leading to a significant relative decrease of up to 90.9%, while in cross-language scenario, the drop reaches 91.3%. Despite our proxy-attack strategy successfully bypassing the detectors with such significant relative drops, we find that the generation quality of the attacked models remains preserved, even within a modest utility budget, when compared to the text produced by the original, unattacked source model.



## **38. Be a Multitude to Itself: A Prompt Evolution Framework for Red Teaming**

cs.CL

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16109v1) [paper-pdf](http://arxiv.org/pdf/2502.16109v1)

**Authors**: Rui Li, Peiyi Wang, Jingyuan Ma, Di Zhang, Lei Sha, Zhifang Sui

**Abstract**: Large Language Models (LLMs) have gained increasing attention for their remarkable capacity, alongside concerns about safety arising from their potential to produce harmful content. Red teaming aims to find prompts that could elicit harmful responses from LLMs, and is essential to discover and mitigate safety risks before real-world deployment. However, manual red teaming is both time-consuming and expensive, rendering it unscalable. In this paper, we propose RTPE, a scalable evolution framework to evolve red teaming prompts across both breadth and depth dimensions, facilitating the automatic generation of numerous high-quality and diverse red teaming prompts. Specifically, in-breadth evolving employs a novel enhanced in-context learning method to create a multitude of quality prompts, whereas in-depth evolving applies customized transformation operations to enhance both content and form of prompts, thereby increasing diversity. Extensive experiments demonstrate that RTPE surpasses existing representative automatic red teaming methods on both attack success rate and diversity. In addition, based on 4,800 red teaming prompts created by RTPE, we further provide a systematic analysis of 8 representative LLMs across 8 sensitive topics.



## **39. Merger-as-a-Stealer: Stealing Targeted PII from Aligned LLMs with Model Merging**

cs.CR

17 pages, 3 figures

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16094v1) [paper-pdf](http://arxiv.org/pdf/2502.16094v1)

**Authors**: Lin Lu, Zhigang Zuo, Ziji Sheng, Pan Zhou

**Abstract**: Model merging has emerged as a promising approach for updating large language models (LLMs) by integrating multiple domain-specific models into a cross-domain merged model. Despite its utility and plug-and-play nature, unmonitored mergers can introduce significant security vulnerabilities, such as backdoor attacks and model merging abuse. In this paper, we identify a novel and more realistic attack surface where a malicious merger can extract targeted personally identifiable information (PII) from an aligned model with model merging. Specifically, we propose \texttt{Merger-as-a-Stealer}, a two-stage framework to achieve this attack: First, the attacker fine-tunes a malicious model to force it to respond to any PII-related queries. The attacker then uploads this malicious model to the model merging conductor and obtains the merged model. Second, the attacker inputs direct PII-related queries to the merged model to extract targeted PII. Extensive experiments demonstrate that \texttt{Merger-as-a-Stealer} successfully executes attacks against various LLMs and model merging methods across diverse settings, highlighting the effectiveness of the proposed framework. Given that this attack enables character-level extraction for targeted PII without requiring any additional knowledge from the attacker, we stress the necessity for improved model alignment and more robust defense mechanisms to mitigate such threats.



## **40. Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack**

cs.CR

12 pages, 5 figures

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2502.16086v1) [paper-pdf](http://arxiv.org/pdf/2502.16086v1)

**Authors**: Chenxi Dai, Lin Lu, Pan Zhou

**Abstract**: Decentralized training has become a resource-efficient framework to democratize the training of large language models (LLMs). However, the privacy risks associated with this framework, particularly due to the potential inclusion of sensitive data in training datasets, remain unexplored. This paper identifies a novel and realistic attack surface: the privacy leakage from training data in decentralized training, and proposes \textit{activation inversion attack} (AIA) for the first time. AIA first constructs a shadow dataset comprising text labels and corresponding activations using public datasets. Leveraging this dataset, an attack model can be trained to reconstruct the training data from activations in victim decentralized training. We conduct extensive experiments on various LLMs and publicly available datasets to demonstrate the susceptibility of decentralized training to AIA. These findings highlight the urgent need to enhance security measures in decentralized training to mitigate privacy risks in training LLMs.



## **41. Understanding the Effectiveness of Coverage Criteria for Large Language Models: A Special Angle from Jailbreak Attacks**

cs.SE

**SubmitDate**: 2025-02-22    [abs](http://arxiv.org/abs/2408.15207v2) [paper-pdf](http://arxiv.org/pdf/2408.15207v2)

**Authors**: Shide Zhou, Tianlin Li, Kailong Wang, Yihao Huang, Ling Shi, Yang Liu, Haoyu Wang

**Abstract**: Large language models (LLMs) have revolutionized artificial intelligence, but their increasing deployment across critical domains has raised concerns about their abnormal behaviors when faced with malicious attacks. Such vulnerability alerts the widespread inadequacy of pre-release testing.In this paper, we conduct a comprehensive empirical study to evaluate the effectiveness of traditional coverage criteria in identifying such inadequacies, exemplified by the significant security concern of jailbreak attacks.Our study begins with a clustering analysis of the hidden states of LLMs, revealing that the embedded characteristics effectively distinguish between different query types. We then systematically evaluate the performance of these criteria across three key dimensions: criterion level, layer level, and token level. Our research uncovers significant differences in neuron coverage when LLMs process normal versus jailbreak queries, aligning with our clustering experiments.Leveraging these findings, we propose three practical applications of coverage criteria in the context of LLM security testing. Specifically, we develop a real-time jailbreak detection mechanism that achieves high accuracy (93.61% on average) in classifying queries as normal or jailbreak. Furthermore, we explore the use of coverage levels to prioritize test cases, improving testing efficiency by focusing on high-risk interactions and removing redundant tests. Lastly, we introduce a coverage-guided approach for generating jailbreak attack examples, enabling systematic refinement of prompts to uncover vulnerabilities. This study improves our understanding of LLM security testing, enhances their safety, and provides a foundation for developing more robust AI applications.



## **42. TurboFuzzLLM: Turbocharging Mutation-based Fuzzing for Effectively Jailbreaking Large Language Models in Practice**

cs.CR

Accepted at NAACL 2025 industry track, 12 pages, 5 figures

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.18504v1) [paper-pdf](http://arxiv.org/pdf/2502.18504v1)

**Authors**: Aman Goel, Xian Carrie Wu, Zhe Wang, Dmitriy Bespalov, Yanjun Qi

**Abstract**: Jailbreaking large-language models (LLMs) involves testing their robustness against adversarial prompts and evaluating their ability to withstand prompt attacks that could elicit unauthorized or malicious responses. In this paper, we present TurboFuzzLLM, a mutation-based fuzzing technique for efficiently finding a collection of effective jailbreaking templates that, when combined with harmful questions, can lead a target LLM to produce harmful responses through black-box access via user prompts. We describe the limitations of directly applying existing template-based attacking techniques in practice, and present functional and efficiency-focused upgrades we added to mutation-based fuzzing to generate effective jailbreaking templates automatically. TurboFuzzLLM achieves $\geq$ 95\% attack success rates (ASR) on public datasets for leading LLMs (including GPT-4o \& GPT-4 Turbo), shows impressive generalizability to unseen harmful questions, and helps in improving model defenses to prompt attacks.



## **43. CVE-LLM : Ontology-Assisted Automatic Vulnerability Evaluation Using Large Language Models**

cs.CL

arXiv admin note: substantial text overlap with arXiv:2407.14640

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15932v1) [paper-pdf](http://arxiv.org/pdf/2502.15932v1)

**Authors**: Rikhiya Ghosh, Hans-Martin von Stockhausen, Martin Schmitt, George Marica Vasile, Sanjeev Kumar Karn, Oladimeji Farri

**Abstract**: The National Vulnerability Database (NVD) publishes over a thousand new vulnerabilities monthly, with a projected 25 percent increase in 2024, highlighting the crucial need for rapid vulnerability identification to mitigate cybersecurity attacks and save costs and resources. In this work, we propose using large language models (LLMs) to learn vulnerability evaluation from historical assessments of medical device vulnerabilities in a single manufacturer's portfolio. We highlight the effectiveness and challenges of using LLMs for automatic vulnerability evaluation and introduce a method to enrich historical data with cybersecurity ontologies, enabling the system to understand new vulnerabilities without retraining the LLM. Our LLM system integrates with the in-house application - Cybersecurity Management System (CSMS) - to help Siemens Healthineers (SHS) product cybersecurity experts efficiently assess the vulnerabilities in our products. Also, we present guidelines for efficient integration of LLMs into the cybersecurity tool.



## **44. Defending Jailbreak Prompts via In-Context Adversarial Game**

cs.LG

EMNLP 2024 Main Paper

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2402.13148v3) [paper-pdf](http://arxiv.org/pdf/2402.13148v3)

**Authors**: Yujun Zhou, Yufei Han, Haomin Zhuang, Kehan Guo, Zhenwen Liang, Hongyan Bao, Xiangliang Zhang

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Moreover, ICAG demonstrates remarkable transferability to other LLMs, indicating its potential as a versatile defense mechanism.



## **45. IPAD: Inverse Prompt for AI Detection -- A Robust and Explainable LLM-Generated Text Detector**

cs.LG

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15902v1) [paper-pdf](http://arxiv.org/pdf/2502.15902v1)

**Authors**: Zheng Chen, Yushi Feng, Changyang He, Yue Deng, Hongxi Pu, Bo Li

**Abstract**: Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinguishing between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide explainable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and a Distinguisher that examines how well the input texts align with the predicted prompts. We develop and examine two versions of Distinguishers. Empirical evaluations demonstrate that both Distinguishers perform significantly better than the baseline methods, with version2 outperforming baselines by 9.73% on in-distribution data (F1-score) and 12.65% on OOD data (AUROC). Furthermore, a user study is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.



## **46. A Comprehensive Survey on the Trustworthiness of Large Language Models in Healthcare**

cs.CY

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15871v1) [paper-pdf](http://arxiv.org/pdf/2502.15871v1)

**Authors**: Manar Aljohani, Jun Hou, Sindhura Kommu, Xuan Wang

**Abstract**: The application of large language models (LLMs) in healthcare has the potential to revolutionize clinical decision-making, medical research, and patient care. As LLMs are increasingly integrated into healthcare systems, several critical challenges must be addressed to ensure their reliable and ethical deployment. These challenges include truthfulness, where models generate misleading information; privacy, with risks of unintentional data retention; robustness, requiring defenses against adversarial attacks; fairness, addressing biases in clinical outcomes; explainability, ensuring transparent decision-making; and safety, mitigating risks of misinformation and medical errors. Recently, researchers have begun developing benchmarks and evaluation frameworks to systematically assess the trustworthiness of LLMs. However, the trustworthiness of LLMs in healthcare remains underexplored, lacking a systematic review that provides a comprehensive understanding and future insights into this area. This survey bridges this gap by providing a comprehensive overview of the recent research of existing methodologies and solutions aimed at mitigating the above risks in healthcare. By focusing on key trustworthiness dimensions including truthfulness, privacy and safety, robustness, fairness and bias, and explainability, we present a thorough analysis of how these issues impact the reliability and ethical use of LLMs in healthcare. This paper highlights ongoing efforts and offers insights into future research directions to ensure the safe and trustworthy deployment of LLMs in healthcare.



## **47. SafeInt: Shielding Large Language Models from Jailbreak Attacks via Safety-Aware Representation Intervention**

cs.CL

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15594v1) [paper-pdf](http://arxiv.org/pdf/2502.15594v1)

**Authors**: Jiaqi Wu, Chen Chen, Chunyan Hou, Xiaojie Yuan

**Abstract**: With the widespread real-world deployment of large language models (LLMs), ensuring their behavior complies with safety standards has become crucial. Jailbreak attacks exploit vulnerabilities in LLMs to induce undesirable behavior, posing a significant threat to LLM safety. Previous defenses often fail to achieve both effectiveness and efficiency simultaneously. Defenses from a representation perspective offer new insights, but existing interventions cannot dynamically adjust representations based on the harmfulness of the queries. To address this limitation while ensuring both effectiveness and efficiency, we propose SafeIntervention (SafeInt), a novel defense method that shields LLMs from jailbreak attacks through safety-aware representation intervention. SafeInt is built on our analysis of the representations of jailbreak samples. It adjusts representation distributions of jailbreak samples through intervention to align them with the representations of unsafe samples while minimizing unnecessary perturbations to jailbreak-irrelevant representations. We conduct comprehensive experiments covering six jailbreak attacks, two jailbreak datasets, and two utility benchmarks. Experimental results demonstrate that SafeInt outperforms all baselines in defending LLMs against jailbreak attacks while largely maintaining utility. Additionally, we evaluate SafeInt against adaptive attacks and verify its effectiveness in mitigating real-time attacks.



## **48. Interpreting and Steering LLMs with Mutual Information-based Explanations on Sparse Autoencoders**

cs.CL

Pre-print. 20 pages, 5 figures

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15576v1) [paper-pdf](http://arxiv.org/pdf/2502.15576v1)

**Authors**: Xuansheng Wu, Jiayi Yuan, Wenlin Yao, Xiaoming Zhai, Ninghao Liu

**Abstract**: Large language models (LLMs) excel at handling human queries, but they can occasionally generate flawed or unexpected responses. Understanding their internal states is crucial for understanding their successes, diagnosing their failures, and refining their capabilities. Although sparse autoencoders (SAEs) have shown promise for interpreting LLM internal representations, limited research has explored how to better explain SAE features, i.e., understanding the semantic meaning of features learned by SAE. Our theoretical analysis reveals that existing explanation methods suffer from the frequency bias issue, where they emphasize linguistic patterns over semantic concepts, while the latter is more critical to steer LLM behaviors. To address this, we propose using a fixed vocabulary set for feature interpretations and designing a mutual information-based objective, aiming to better capture the semantic meaning behind these features. We further propose two runtime steering strategies that adjust the learned feature activations based on their corresponding explanations. Empirical results show that, compared to baselines, our method provides more discourse-level explanations and effectively steers LLM behaviors to defend against jailbreak attacks. These findings highlight the value of explanations for steering LLM behaviors in downstream applications. We will release our code and data once accepted.



## **49. Construction and Evaluation of LLM-based agents for Semi-Autonomous penetration testing**

cs.CR

7 pages, 4 tables and 1 figure

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15506v1) [paper-pdf](http://arxiv.org/pdf/2502.15506v1)

**Authors**: Masaya Kobayashi, Masane Fuchi, Amar Zanashir, Tomonori Yoneda, Tomohiro Takagi

**Abstract**: With the emergence of high-performance large language models (LLMs) such as GPT, Claude, and Gemini, the autonomous and semi-autonomous execution of tasks has significantly advanced across various domains. However, in highly specialized fields such as cybersecurity, full autonomy remains a challenge. This difficulty primarily stems from the limitations of LLMs in reasoning capabilities and domain-specific knowledge. We propose a system that semi-autonomously executes complex cybersecurity workflows by employing multiple LLMs modules to formulate attack strategies, generate commands, and analyze results, thereby addressing the aforementioned challenges. In our experiments using Hack The Box virtual machines, we confirmed that our system can autonomously construct attack strategies, issue appropriate commands, and automate certain processes, thereby reducing the need for manual intervention.



## **50. Single-pass Detection of Jailbreaking Input in Large Language Models**

cs.LG

Accepted in TMLR 2025

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.15435v1) [paper-pdf](http://arxiv.org/pdf/2502.15435v1)

**Authors**: Leyla Naz Candogan, Yongtao Wu, Elias Abad Rocamora, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Defending aligned Large Language Models (LLMs) against jailbreaking attacks is a challenging problem, with existing approaches requiring multiple requests or even queries to auxiliary LLMs, making them computationally heavy. Instead, we focus on detecting jailbreaking input in a single forward pass. Our method, called Single Pass Detection SPD, leverages the information carried by the logits to predict whether the output sentence will be harmful. This allows us to defend in just one forward pass. SPD can not only detect attacks effectively on open-source models, but also minimizes the misclassification of harmless inputs. Furthermore, we show that SPD remains effective even without complete logit access in GPT-3.5 and GPT-4. We believe that our proposed method offers a promising approach to efficiently safeguard LLMs against adversarial attacks.



