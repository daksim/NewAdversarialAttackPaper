# Latest Large Language Model Attack Papers
**update at 2024-12-04 10:55:44**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Gracefully Filtering Backdoor Samples for Generative Large Language Models without Retraining**

cs.CL

Accepted at COLING 2025

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02454v1) [paper-pdf](http://arxiv.org/pdf/2412.02454v1)

**Authors**: Zongru Wu, Pengzhou Cheng, Lingyong Fang, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Backdoor attacks remain significant security threats to generative large language models (LLMs). Since generative LLMs output sequences of high-dimensional token logits instead of low-dimensional classification logits, most existing backdoor defense methods designed for discriminative models like BERT are ineffective for generative LLMs. Inspired by the observed differences in learning behavior between backdoor and clean mapping in the frequency space, we transform gradients of each training sample, directly influencing parameter updates, into the frequency space. Our findings reveal a distinct separation between the gradients of backdoor and clean samples in the frequency space. Based on this phenomenon, we propose Gradient Clustering in the Frequency Space for Backdoor Sample Filtering (GraCeFul), which leverages sample-wise gradients in the frequency space to effectively identify backdoor samples without requiring retraining LLMs. Experimental results show that GraCeFul outperforms baselines significantly. Notably, GraCeFul exhibits remarkable computational efficiency, achieving nearly 100% recall and F1 scores in identifying backdoor samples, reducing the average success rate of various backdoor attacks to 0% with negligible drops in clean accuracy across multiple free-style question answering datasets. Additionally, GraCeFul generalizes to Llama-2 and Vicuna. The codes are publicly available at https://github.com/ZrW00/GraceFul.



## **2. Jailbreak Large Vision-Language Models Through Multi-Modal Linkage**

cs.CV

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.00473v2) [paper-pdf](http://arxiv.org/pdf/2412.00473v2)

**Authors**: Yu Wang, Xiaofei Zhou, Yichen Wang, Geyuan Zhang, Tianxing He

**Abstract**: With the significant advancement of Large Vision-Language Models (VLMs), concerns about their potential misuse and abuse have grown rapidly. Previous studies have highlighted VLMs' vulnerability to jailbreak attacks, where carefully crafted inputs can lead the model to produce content that violates ethical and legal standards. However, existing methods struggle against state-of-the-art VLMs like GPT-4o, due to the over-exposure of harmful content and lack of stealthy malicious guidance. In this work, we propose a novel jailbreak attack framework: Multi-Modal Linkage (MML) Attack. Drawing inspiration from cryptography, MML utilizes an encryption-decryption process across text and image modalities to mitigate over-exposure of malicious information. To align the model's output with malicious intent covertly, MML employs a technique called "evil alignment", framing the attack within a video game production scenario. Comprehensive experiments demonstrate MML's effectiveness. Specifically, MML jailbreaks GPT-4o with attack success rates of 97.80% on SafeBench, 98.81% on MM-SafeBench and 99.07% on HADES-Dataset. Our code is available at https://github.com/wangyu-ovo/MML



## **3. Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2409.18169v5) [paper-pdf](http://arxiv.org/pdf/2409.18169v5)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent research demonstrates that the nascent fine-tuning-as-a-service business model exposes serious safety concerns -- fine-tuning over a few harmful data uploaded by the users can compromise the safety alignment of the model. The attack, known as harmful fine-tuning attack, has raised a broad research interest among the community. However, as the attack is still new, \textbf{we observe that there are general misunderstandings within the research community.} To clear up concern, this paper provide a comprehensive overview to three aspects of harmful fine-tuning: attacks setting, defense design and evaluation methodology. Specifically, we first present the threat model of the problem, and introduce the harmful fine-tuning attack and its variants. Then we systematically survey the existing literature on attacks/defenses/mechanical analysis of the problem. Finally, we introduce the evaluation methodology and outline future research directions that might contribute to the development of the field. Additionally, we present a list of questions of interest, which might be useful to refer to when reviewers in the peer review process question the realism of the experiment/attack/defense setting. A curated list of relevant papers is maintained and made accessible at: https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers.



## **4. Trust & Safety of LLMs and LLMs in Trust & Safety**

cs.AI

11 pages

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02113v1) [paper-pdf](http://arxiv.org/pdf/2412.02113v1)

**Authors**: Doohee You, Dan Chon

**Abstract**: In recent years, Large Language Models (LLMs) have garnered considerable attention for their remarkable abilities in natural language processing tasks. However, their widespread adoption has raised concerns pertaining to trust and safety. This systematic review investigates the current research landscape on trust and safety in LLMs, with a particular focus on the novel application of LLMs within the field of Trust and Safety itself. We delve into the complexities of utilizing LLMs in domains where maintaining trust and safety is paramount, offering a consolidated perspective on this emerging trend.\   By synthesizing findings from various studies, we identify key challenges and potential solutions, aiming to benefit researchers and practitioners seeking to understand the nuanced interplay between LLMs and Trust and Safety.   This review provides insights on best practices for using LLMs in Trust and Safety, and explores emerging risks such as prompt injection and jailbreak attacks. Ultimately, this study contributes to a deeper understanding of how LLMs can be effectively and responsibly utilized to enhance trust and safety in the digital realm.



## **5. Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders**

cs.CL

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2411.07870v3) [paper-pdf](http://arxiv.org/pdf/2411.07870v3)

**Authors**: Xiaofeng Zhu, Jaya Krishna Mandivarapu

**Abstract**: Although people are impressed by the content generation skills of large language models, the use of LLMs, such as ChatGPT, is limited by the domain grounding of the content. The correctness and groundedness of the generated content need to be based on a verified context, such as results from Retrieval-Augmented Generation (RAG). One important issue when adapting LLMs to a customized domain is that the generated responses are often incomplete, or the additions are not verified and may even be hallucinated. Prior studies on hallucination detection have focused on evaluation metrics, which are not easily adaptable to dynamic domains and can be vulnerable to attacks like jail-breaking. In this work, we propose 1) a post-processing algorithm that leverages knowledge triplets in RAG context to correct hallucinations and 2) a dual-decoder model that fuses RAG context to guide the generation process.



## **6. Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis**

cs.CL

Accepted by EMNLP 2024 Main

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2406.10794v3) [paper-pdf](http://arxiv.org/pdf/2406.10794v3)

**Authors**: Yuping Lin, Pengfei He, Han Xu, Yue Xing, Makoto Yamada, Hui Liu, Jiliang Tang

**Abstract**: Large language models (LLMs) are susceptible to a type of attack known as jailbreaking, which misleads LLMs to output harmful contents. Although there are diverse jailbreak attack strategies, there is no unified understanding on why some methods succeed and others fail. This paper explores the behavior of harmful and harmless prompts in the LLM's representation space to investigate the intrinsic properties of successful jailbreak attacks. We hypothesize that successful attacks share some similar properties: They are effective in moving the representation of the harmful prompt towards the direction to the harmless prompts. We leverage hidden representations into the objective of existing jailbreak attacks to move the attacks along the acceptance direction, and conduct experiments to validate the above hypothesis using the proposed objective. We hope this study provides new insights into understanding how LLMs understand harmfulness information.



## **7. Improved Large Language Model Jailbreak Detection via Pretrained Embeddings**

cs.CR

Submitted to AICS 2025: https://aics.site

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01547v1) [paper-pdf](http://arxiv.org/pdf/2412.01547v1)

**Authors**: Erick Galinkin, Martin Sablotny

**Abstract**: The adoption of large language models (LLMs) in many applications, from customer service chat bots and software development assistants to more capable agentic systems necessitates research into how to secure these systems. Attacks like prompt injection and jailbreaking attempt to elicit responses and actions from these models that are not compliant with the safety, privacy, or content policies of organizations using the model in their application. In order to counter abuse of LLMs for generating potentially harmful replies or taking undesirable actions, LLM owners must apply safeguards during training and integrate additional tools to block the LLM from generating text that abuses the model. Jailbreaking prompts play a vital role in convincing an LLM to generate potentially harmful content, making it important to identify jailbreaking attempts to block any further steps. In this work, we propose a novel approach to detect jailbreak prompts based on pairing text embeddings well-suited for retrieval with traditional machine learning classification algorithms. Our approach outperforms all publicly available methods from open source LLM security applications.



## **8. LUMIA: Linear probing for Unimodal and MultiModal Membership Inference Attacks leveraging internal LLM states**

cs.CR

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2411.19876v2) [paper-pdf](http://arxiv.org/pdf/2411.19876v2)

**Authors**: Luis Ibanez-Lissen, Lorena Gonzalez-Manzano, Jose Maria de Fuentes, Nicolas Anciaux, Joaquin Garcia-Alfaro

**Abstract**: Large Language Models (LLMs) are increasingly used in a variety of applications, but concerns around membership inference have grown in parallel. Previous efforts focus on black-to-grey-box models, thus neglecting the potential benefit from internal LLM information. To address this, we propose the use of Linear Probes (LPs) as a method to detect Membership Inference Attacks (MIAs) by examining internal activations of LLMs. Our approach, dubbed LUMIA, applies LPs layer-by-layer to get fine-grained data on the model inner workings. We test this method across several model architectures, sizes and datasets, including unimodal and multimodal tasks. In unimodal MIA, LUMIA achieves an average gain of 15.71 % in Area Under the Curve (AUC) over previous techniques. Remarkably, LUMIA reaches AUC>60% in 65.33% of cases -- an increment of 46.80% against the state of the art. Furthermore, our approach reveals key insights, such as the model layers where MIAs are most detectable. In multimodal models, LPs indicate that visual inputs can significantly contribute to detect MIAs -- AUC>60% is reached in 85.90% of experiments.



## **9. Recent Advances in Attack and Defense Approaches of Large Language Models**

cs.CR

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2409.03274v3) [paper-pdf](http://arxiv.org/pdf/2409.03274v3)

**Authors**: Jing Cui, Yishi Xu, Zhewei Huang, Shuchang Zhou, Jianbin Jiao, Junge Zhang

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence and machine learning through their advanced text processing and generating capabilities. However, their widespread deployment has raised significant safety and reliability concerns. Established vulnerabilities in deep neural networks, coupled with emerging threat models, may compromise security evaluations and create a false sense of security. Given the extensive research in the field of LLM security, we believe that summarizing the current state of affairs will help the research community better understand the present landscape and inform future developments. This paper reviews current research on LLM vulnerabilities and threats, and evaluates the effectiveness of contemporary defense mechanisms. We analyze recent studies on attack vectors and model weaknesses, providing insights into attack mechanisms and the evolving threat landscape. We also examine current defense strategies, highlighting their strengths and limitations. By contrasting advancements in attack and defense methodologies, we identify research gaps and propose future directions to enhance LLM security. Our goal is to advance the understanding of LLM safety challenges and guide the development of more robust security measures.



## **10. BDefects4NN: A Backdoor Defect Database for Controlled Localization Studies in Neural Networks**

cs.SE

11 pages, accepted by ICSE 2025

**SubmitDate**: 2024-12-01    [abs](http://arxiv.org/abs/2412.00746v1) [paper-pdf](http://arxiv.org/pdf/2412.00746v1)

**Authors**: Yisong Xiao, Aishan Liu, Xinwei Zhang, Tianyuan Zhang, Tianlin Li, Siyuan Liang, Xianglong Liu, Yang Liu, Dacheng Tao

**Abstract**: Pre-trained large deep learning models are now serving as the dominant component for downstream middleware users and have revolutionized the learning paradigm, replacing the traditional approach of training from scratch locally. To reduce development costs, developers often integrate third-party pre-trained deep neural networks (DNNs) into their intelligent software systems. However, utilizing untrusted DNNs presents significant security risks, as these models may contain intentional backdoor defects resulting from the black-box training process. These backdoor defects can be activated by hidden triggers, allowing attackers to maliciously control the model and compromise the overall reliability of the intelligent software. To ensure the safe adoption of DNNs in critical software systems, it is crucial to establish a backdoor defect database for localization studies. This paper addresses this research gap by introducing BDefects4NN, the first backdoor defect database, which provides labeled backdoor-defected DNNs at the neuron granularity and enables controlled localization studies of defect root causes. In BDefects4NN, we define three defect injection rules and employ four representative backdoor attacks across four popular network architectures and three widely adopted datasets, yielding a comprehensive database of 1,654 backdoor-defected DNNs with four defect quantities and varying infected neurons. Based on BDefects4NN, we conduct extensive experiments on evaluating six fault localization criteria and two defect repair techniques, which show limited effectiveness for backdoor defects. Additionally, we investigate backdoor-defected models in practical scenarios, specifically in lane detection for autonomous driving and large language models (LLMs), revealing potential threats and highlighting current limitations in precise defect localization.



## **11. Evaluating Large Language Models' Capability to Launch Fully Automated Spear Phishing Campaigns: Validated on Human Subjects**

cs.CR

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2412.00586v1) [paper-pdf](http://arxiv.org/pdf/2412.00586v1)

**Authors**: Fred Heiding, Simon Lermen, Andrew Kao, Bruce Schneier, Arun Vishwanath

**Abstract**: In this paper, we evaluate the capability of large language models to conduct personalized phishing attacks and compare their performance with human experts and AI models from last year. We include four email groups with a combined total of 101 participants: A control group of arbitrary phishing emails, which received a click-through rate (recipient pressed a link in the email) of 12%, emails generated by human experts (54% click-through), fully AI-automated emails 54% (click-through), and AI emails utilizing a human-in-the-loop (56% click-through). Thus, the AI-automated attacks performed on par with human experts and 350% better than the control group. The results are a significant improvement from similar studies conducted last year, highlighting the increased deceptive capabilities of AI models. Our AI-automated emails were sent using a custom-built tool that automates the entire spear phishing process, including information gathering and creating personalized vulnerability profiles for each target. The AI-gathered information was accurate and useful in 88% of cases and only produced inaccurate profiles for 4% of the participants. We also use language models to detect the intention of emails. Claude 3.5 Sonnet scored well above 90% with low false-positive rates and detected several seemingly benign emails that passed human detection. Lastly, we analyze the economics of phishing, highlighting how AI enables attackers to target more individuals at lower cost and increase profitability by up to 50 times for larger audiences.



## **12. Uncovering Safety Risks of Large Language Models through Concept Activation Vector**

cs.CL

10 pages, accepted at NeurIPS 2024

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2404.12038v5) [paper-pdf](http://arxiv.org/pdf/2404.12038v5)

**Authors**: Zhihao Xu, Ruixuan Huang, Changyu Chen, Xiting Wang

**Abstract**: Despite careful safety alignment, current large language models (LLMs) remain vulnerable to various attacks. To further unveil the safety risks of LLMs, we introduce a Safety Concept Activation Vector (SCAV) framework, which effectively guides the attacks by accurately interpreting LLMs' safety mechanisms. We then develop an SCAV-guided attack method that can generate both attack prompts and embedding-level attacks with automatically selected perturbation hyperparameters. Both automatic and human evaluations demonstrate that our attack method significantly improves the attack success rate and response quality while requiring less training data. Additionally, we find that our generated attack prompts may be transferable to GPT-4, and the embedding-level attacks may also be transferred to other white-box LLMs whose parameters are known. Our experiments further uncover the safety risks present in current LLMs. For example, in our evaluation of seven open-source LLMs, we observe an average attack success rate of 99.14%, based on the classic keyword-matching criterion. Finally, we provide insights into the safety mechanism of LLMs. The code is available at https://github.com/SproutNan/AI-Safety_SCAV.



## **13. Safety Alignment Backfires: Preventing the Re-emergence of Suppressed Concepts in Fine-tuned Text-to-Image Diffusion Models**

cs.AI

20 pages, 18 figures

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2412.00357v1) [paper-pdf](http://arxiv.org/pdf/2412.00357v1)

**Authors**: Sanghyun Kim, Moonseok Choi, Jinwoo Shin, Juho Lee

**Abstract**: Fine-tuning text-to-image diffusion models is widely used for personalization and adaptation for new domains. In this paper, we identify a critical vulnerability of fine-tuning: safety alignment methods designed to filter harmful content (e.g., nudity) can break down during fine-tuning, allowing previously suppressed content to resurface, even when using benign datasets. While this "fine-tuning jailbreaking" issue is known in large language models, it remains largely unexplored in text-to-image diffusion models. Our investigation reveals that standard fine-tuning can inadvertently undo safety measures, causing models to relearn harmful concepts that were previously removed and even exacerbate harmful behaviors. To address this issue, we present a novel but immediate solution called Modular LoRA, which involves training Safety Low-Rank Adaptation (LoRA) modules separately from Fine-Tuning LoRA components and merging them during inference. This method effectively prevents the re-learning of harmful content without compromising the model's performance on new tasks. Our experiments demonstrate that Modular LoRA outperforms traditional fine-tuning methods in maintaining safety alignment, offering a practical approach for enhancing the security of text-to-image diffusion models against potential attacks.



## **14. When LLMs Go Online: The Emerging Threat of Web-Enabled LLMs**

cs.CR

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2410.14569v2) [paper-pdf](http://arxiv.org/pdf/2410.14569v2)

**Authors**: Hanna Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin, Kimin Lee

**Abstract**: Recent advancements in Large Language Models (LLMs) have established them as agentic systems capable of planning and interacting with various tools. These LLM agents are often paired with web-based tools, enabling access to diverse sources and real-time information. Although these advancements offer significant benefits across various applications, they also increase the risk of malicious use, particularly in cyberattacks involving personal information. In this work, we investigate the risks associated with misuse of LLM agents in cyberattacks involving personal data. Specifically, we aim to understand: 1) how potent LLM agents can be when directed to conduct cyberattacks, 2) how cyberattacks are enhanced by web-based tools, and 3) how affordable and easy it becomes to launch cyberattacks using LLM agents. We examine three attack scenarios: the collection of Personally Identifiable Information (PII), the generation of impersonation posts, and the creation of spear-phishing emails. Our experiments reveal the effectiveness of LLM agents in these attacks: LLM agents achieved a precision of up to 95.9% in collecting PII, up to 93.9% of impersonation posts created by LLM agents were evaluated as authentic, and the click rate for links in spear phishing emails created by LLM agents reached up to 46.67%. Additionally, our findings underscore the limitations of existing safeguards in contemporary commercial LLMs, emphasizing the urgent need for more robust security measures to prevent the misuse of LLM agents.



## **15. Ensemble Watermarks for Large Language Models**

cs.CL

9 pages in the main body. Code is available at  http://github.com/CommodoreEU/master-generation. arXiv admin note:  substantial text overlap with arXiv:2405.08400

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2411.19563v1) [paper-pdf](http://arxiv.org/pdf/2411.19563v1)

**Authors**: Georg Niess, Roman Kern

**Abstract**: The rapid advancement of large language models (LLMs) has made it increasingly difficult to distinguish between text written by humans and machines. While watermarks already exist for LLMs, they often lack flexibility, and struggle with attacks such as paraphrasing. To address these issues, we propose a multi-feature method for generating watermarks that combines multiple distinct watermark features into an ensemble watermark. Concretely, we combine acrostica and sensorimotor norms with the established red-green watermark to achieve a 98% detection rate. After a paraphrasing attack the performance remains high with 95% detection rate. The red-green feature alone as baseline achieves a detection rate of 49%. The evaluation of all feature combinations reveals that the ensemble of all three consistently has the highest detection rate across several LLMs and watermark strength settings. Due to the flexibility of combining features in the ensemble, various requirements and trade-offs can be addressed. Additionally, for all ensemble configurations the same detection function can be used without adaptations. This method is particularly of interest to facilitate accountability and prevent societal harm.



## **16. InputSnatch: Stealing Input in LLM Services via Timing Side-Channel Attacks**

cs.CR

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2411.18191v2) [paper-pdf](http://arxiv.org/pdf/2411.18191v2)

**Authors**: Xinyao Zheng, Husheng Han, Shangyi Shi, Qiyan Fang, Zidong Du, Xing Hu, Qi Guo

**Abstract**: Large language models (LLMs) possess extensive knowledge and question-answering capabilities, having been widely deployed in privacy-sensitive domains like finance and medical consultation. During LLM inferences, cache-sharing methods are commonly employed to enhance efficiency by reusing cached states or responses for the same or similar inference requests. However, we identify that these cache mechanisms pose a risk of private input leakage, as the caching can result in observable variations in response times, making them a strong candidate for a timing-based attack hint.   In this study, we propose a novel timing-based side-channel attack to execute input theft in LLMs inference. The cache-based attack faces the challenge of constructing candidate inputs in a large search space to hit and steal cached user queries. To address these challenges, we propose two primary components. The input constructor employs machine learning techniques and LLM-based approaches for vocabulary correlation learning while implementing optimized search mechanisms for generalized input construction. The time analyzer implements statistical time fitting with outlier elimination to identify cache hit patterns, continuously providing feedback to refine the constructor's search strategy. We conduct experiments across two cache mechanisms and the results demonstrate that our approach consistently attains high attack success rates in various applications. Our work highlights the security vulnerabilities associated with performance optimizations, underscoring the necessity of prioritizing privacy and security alongside enhancements in LLM inference.



## **17. RePD: Defending Jailbreak Attack through a Retrieval-based Prompt Decomposition Process**

cs.CR

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2410.08660v3) [paper-pdf](http://arxiv.org/pdf/2410.08660v3)

**Authors**: Peiran Wang, Xiaogeng Liu, Chaowei Xiao

**Abstract**: In this study, we introduce RePD, an innovative attack Retrieval-based Prompt Decomposition framework designed to mitigate the risk of jailbreak attacks on large language models (LLMs). Despite rigorous pretraining and finetuning focused on ethical alignment, LLMs are still susceptible to jailbreak exploits. RePD operates on a one-shot learning model, wherein it accesses a database of pre-collected jailbreak prompt templates to identify and decompose harmful inquiries embedded within user prompts. This process involves integrating the decomposition of the jailbreak prompt into the user's original query into a one-shot learning example to effectively teach the LLM to discern and separate malicious components. Consequently, the LLM is equipped to first neutralize any potentially harmful elements before addressing the user's prompt in a manner that aligns with its ethical guidelines. RePD is versatile and compatible with a variety of open-source LLMs acting as agents. Through comprehensive experimentation with both harmful and benign prompts, we have demonstrated the efficacy of our proposed RePD in enhancing the resilience of LLMs against jailbreak attacks, without compromising their performance in responding to typical user requests.



## **18. Confidential Prompting: Protecting User Prompts from Cloud LLM Providers**

cs.CR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2409.19134v2) [paper-pdf](http://arxiv.org/pdf/2409.19134v2)

**Authors**: In Gim, Caihua Li, Lin Zhong

**Abstract**: Our work tackles the challenge of securing user inputs in cloud-hosted large language model (LLM) serving while ensuring output invariance, model confidentiality, and compute efficiency. We introduce secure multi-party decoding (SMD), which leverages confidential computing to confine user prompts to a trusted execution environment (TEE), namely a confidential virtual machine (CVM), while allowing service providers to generate tokens efficiently. We also introduce a novel cryptographic method, prompt obfuscation (PO), to ensure robustness against reconstruction attacks on SMD. We demonstrate that our approach preserves both prompt confidentiality and LLM serving efficiency. Our solution can enable privacy-preserving cloud LLM serving that handles sensitive prompts, such as clinical records, financial data, and personal information.



## **19. Memorization of Named Entities in Fine-tuned BERT Models**

cs.CL

published at CD-MAKE 2023

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2212.03749v3) [paper-pdf](http://arxiv.org/pdf/2212.03749v3)

**Authors**: Andor Diera, Nicolas Lell, Aygul Garifullina, Ansgar Scherp

**Abstract**: Privacy preserving deep learning is an emerging field in machine learning that aims to mitigate the privacy risks in the use of deep neural networks. One such risk is training data extraction from language models that have been trained on datasets, which contain personal and privacy sensitive information. In our study, we investigate the extent of named entity memorization in fine-tuned BERT models. We use single-label text classification as representative downstream task and employ three different fine-tuning setups in our experiments, including one with Differential Privacy (DP). We create a large number of text samples from the fine-tuned BERT models utilizing a custom sequential sampling strategy with two prompting strategies. We search in these samples for named entities and check if they are also present in the fine-tuning datasets. We experiment with two benchmark datasets in the domains of emails and blogs. We show that the application of DP has a detrimental effect on the text generation capabilities of BERT. Furthermore, we show that a fine-tuned BERT does not generate more named entities specific to the fine-tuning dataset than a BERT model that is pre-trained only. This suggests that BERT is unlikely to emit personal or privacy sensitive named entities. Overall, our results are important to understand to what extent BERT-based services are prone to training data extraction attacks.



## **20. On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks**

cs.CR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2407.04794v2) [paper-pdf](http://arxiv.org/pdf/2407.04794v2)

**Authors**: Zesen Liu, Tianshuo Cong, Xinlei He, Qi Li

**Abstract**: Large Language Models (LLMs) excel in various applications, including text generation and complex tasks. However, the misuse of LLMs raises concerns about the authenticity and ethical implications of the content they produce, such as deepfake news, academic fraud, and copyright infringement. Watermarking techniques, which embed identifiable markers in machine-generated text, offer a promising solution to these issues by allowing for content verification and origin tracing. Unfortunately, the robustness of current LLM watermarking schemes under potential watermark removal attacks has not been comprehensively explored.   In this paper, to fill this gap, we first systematically comb the mainstream watermarking schemes and removal attacks on machine-generated texts, and then we categorize them into pre-text (before text generation) and post-text (after text generation) classes so that we can conduct diversified analyses. In our experiments, we evaluate eight watermarks (five pre-text, three post-text) and twelve attacks (two pre-text, ten post-text) across 87 scenarios. Evaluation results indicate that (1) KGW and Exponential watermarks offer high text quality and watermark retention but remain vulnerable to most attacks; (2) Post-text attacks are found to be more efficient and practical than pre-text attacks; (3) Pre-text watermarks are generally more imperceptible, as they do not alter text fluency, unlike post-text watermarks; (4) Additionally, combined attack methods can significantly increase effectiveness, highlighting the need for more robust watermarking solutions. Our study underscores the vulnerabilities of current techniques and the necessity for developing more resilient schemes.



## **21. Assessing biomedical knowledge robustness in large language models by query-efficient sampling attacks**

cs.CL

31 pages incl. appendix, accepted by TMLR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2402.10527v3) [paper-pdf](http://arxiv.org/pdf/2402.10527v3)

**Authors**: R. Patrick Xian, Alex J. Lee, Satvik Lolla, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. Understanding model vulnerabilities in high-stakes and knowledge-intensive tasks is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples (i.e. adversarial entities) in natural language processing tasks raises questions about their potential impact on the knowledge robustness of pre-trained and finetuned LLMs in high-stakes and specialized domains. We examined the use of type-consistent entity substitution as a template for collecting adversarial entities for billion-parameter LLMs with biomedical knowledge. To this end, we developed an embedding-space attack based on powerscaled distance-weighted sampling to assess the robustness of their biomedical knowledge with a low query budget and controllable coverage. Our method has favorable query efficiency and scaling over alternative approaches based on random sampling and blackbox gradient-guided search, which we demonstrated for adversarial distractor generation in biomedical question answering. Subsequent failure mode analysis uncovered two regimes of adversarial entities on the attack surface with distinct characteristics and we showed that entity substitution attacks can manipulate token-wise Shapley value explanations, which become deceptive in this setting. Our approach complements standard evaluations for high-capacity models and the results highlight the brittleness of domain knowledge in LLMs.



## **22. Knowledge Database or Poison Base? Detecting RAG Poisoning Attack through LLM Activations**

cs.CR

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2411.18948v1) [paper-pdf](http://arxiv.org/pdf/2411.18948v1)

**Authors**: Xue Tan, Hao Luan, Mingyu Luo, Xiaoyan Sun, Ping Chen, Jun Dai

**Abstract**: As Large Language Models (LLMs) are progressively deployed across diverse fields and real-world applications, ensuring the security and robustness of LLMs has become ever more critical. Retrieval-Augmented Generation (RAG) is a cutting-edge approach designed to address the limitations of large language models (LLMs). By retrieving information from the relevant knowledge database, RAG enriches the input to LLMs, enabling them to produce responses that are more accurate and contextually appropriate. It is worth noting that the knowledge database, being sourced from publicly available channels such as Wikipedia, inevitably introduces a new attack surface. RAG poisoning involves injecting malicious texts into the knowledge database, ultimately leading to the generation of the attacker's target response (also called poisoned response). However, there are currently limited methods available for detecting such poisoning attacks. We aim to bridge the gap in this work. Particularly, we introduce RevPRAG, a flexible and automated detection pipeline that leverages the activations of LLMs for poisoned response detection. Our investigation uncovers distinct patterns in LLMs' activations when generating correct responses versus poisoned responses. Our results on multiple benchmark datasets and RAG architectures show our approach could achieve 98% true positive rate, while maintaining false positive rates close to 1%. We also evaluate recent backdoor detection methods specifically designed for LLMs and applicable for identifying poisoned responses in RAG. The results demonstrate that our approach significantly surpasses them.



## **23. SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments**

cs.CV

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2412.00114v1) [paper-pdf](http://arxiv.org/pdf/2412.00114v1)

**Authors**: Yue Cao, Yun Xing, Jie Zhang, Di Lin, Tianwei Zhang, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: Large vision-language models (LVLMs) have shown remarkable capabilities in interpreting visual content. While existing works demonstrate these models' vulnerability to deliberately placed adversarial texts, such texts are often easily identifiable as anomalous. In this paper, we present the first approach to generate scene-coherent typographic adversarial attacks that mislead advanced LVLMs while maintaining visual naturalness through the capability of the LLM-based agent. Our approach addresses three critical questions: what adversarial text to generate, where to place it within the scene, and how to integrate it seamlessly. We propose a training-free, multi-modal LLM-driven scene-coherent typographic adversarial planning (SceneTAP) that employs a three-stage process: scene understanding, adversarial planning, and seamless integration. The SceneTAP utilizes chain-of-thought reasoning to comprehend the scene, formulate effective adversarial text, strategically plan its placement, and provide detailed instructions for natural integration within the image. This is followed by a scene-coherent TextDiffuser that executes the attack using a local diffusion mechanism. We extend our method to real-world scenarios by printing and placing generated patches in physical environments, demonstrating its practical implications. Extensive experiments show that our scene-coherent adversarial text successfully misleads state-of-the-art LVLMs, including ChatGPT-4o, even after capturing new images of physical setups. Our evaluations demonstrate a significant increase in attack success rates while maintaining visual naturalness and contextual appropriateness. This work highlights vulnerabilities in current vision-language models to sophisticated, scene-coherent adversarial attacks and provides insights into potential defense mechanisms.



## **24. Cyber-Attack Technique Classification Using Two-Stage Trained Large Language Models**

cs.LG

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18755v1) [paper-pdf](http://arxiv.org/pdf/2411.18755v1)

**Authors**: Weiqiu You, Youngja Park

**Abstract**: Understanding the attack patterns associated with a cyberattack is crucial for comprehending the attacker's behaviors and implementing the right mitigation measures. However, majority of the information regarding new attacks is typically presented in unstructured text, posing significant challenges for security analysts in collecting necessary information. In this paper, we present a sentence classification system that can identify the attack techniques described in natural language sentences from cyber threat intelligence (CTI) reports. We propose a new method for utilizing auxiliary data with the same labels to improve classification for the low-resource cyberattack classification task. The system first trains the model using the augmented training data and then trains more using only the primary data. We validate our model using the TRAM data1 and the MITRE ATT&CK framework. Experiments show that our method enhances Macro-F1 by 5 to 9 percentage points and keeps Micro-F1 scores competitive when compared to the baseline performance on the TRAM dataset.



## **25. "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks**

cs.CR

This paper has been submitted to Nature Machine Intelligence and  OpenReview preprints. It has 9 pages of text, 3 figures, and 3 tables

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.16730v2) [paper-pdf](http://arxiv.org/pdf/2411.16730v2)

**Authors**: Libo Wang

**Abstract**: As the application of large language models continues to expand in various fields, it poses higher challenges to the effectiveness of identifying harmful content generation and guardrail mechanisms. This research aims to evaluate the guardrail effectiveness of GPT-4o, Grok-2 Beta, Llama 3.1 (405B), Gemini 1.5, and Claude 3.5 Sonnet through black-box testing of seemingly ethical multi-step jailbreak prompts. It conducts ethical attacks by designing an identical multi-step prompts that simulates the scenario of "corporate middle managers competing for promotions." The data results show that the guardrails of the above-mentioned LLMs were bypassed and the content of verbal attacks was generated. Claude 3.5 Sonnet's resistance to multi-step jailbreak prompts is more obvious. To ensure objectivity, the experimental process, black box test code, and enhanced guardrail code are uploaded to the GitHub repository: https://github.com/brucewang123456789/GeniusTrail.git.



## **26. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment**

cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18688v1) [paper-pdf](http://arxiv.org/pdf/2411.18688v1)

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Ahmad Beirami, Furong Huang, Alvaro Velasquez, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks: carefully crafted image-prompt pairs that compel the model to generate harmful content. In this work, we first highlight a critical safety gap, demonstrating that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model during decoding to defend against jailbreak attacks. Additionally, we provide a rigorous mathematical characterization of Immune, offering provable guarantees against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.



## **27. Neutralizing Backdoors through Information Conflicts for Large Language Models**

cs.CL

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18280v1) [paper-pdf](http://arxiv.org/pdf/2411.18280v1)

**Authors**: Chen Chen, Yuchen Sun, Xueluan Gong, Jiaxin Gao, Kwok-Yan Lam

**Abstract**: Large language models (LLMs) have seen significant advancements, achieving superior performance in various Natural Language Processing (NLP) tasks, from understanding to reasoning. However, they remain vulnerable to backdoor attacks, where models behave normally for standard queries but generate harmful responses or unintended output when specific triggers are activated. Existing backdoor defenses often suffer from drawbacks that they either focus on detection without removal, rely on rigid assumptions about trigger properties, or prove to be ineffective against advanced attacks like multi-trigger backdoors. In this paper, we present a novel method to eliminate backdoor behaviors from LLMs through the construction of information conflicts using both internal and external mechanisms. Internally, we leverage a lightweight dataset to train a conflict model, which is then merged with the backdoored model to neutralize malicious behaviors by embedding contradictory information within the model's parametric memory. Externally, we incorporate convincing contradictory evidence into the prompt to challenge the model's internal backdoor knowledge. Experimental results on classification and conversational tasks across 4 widely used LLMs demonstrate that our method outperforms 8 state-of-the-art backdoor defense baselines. We can reduce the attack success rate of advanced backdoor attacks by up to 98% while maintaining over 90% clean data accuracy. Furthermore, our method has proven to be robust against adaptive backdoor attacks. The code will be open-sourced upon publication.



## **28. Visual Adversarial Attack on Vision-Language Models for Autonomous Driving**

cs.CV

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18275v1) [paper-pdf](http://arxiv.org/pdf/2411.18275v1)

**Authors**: Tianyuan Zhang, Lu Wang, Xinwei Zhang, Yitong Zhang, Boyi Jia, Siyuan Liang, Shengshan Hu, Qiang Fu, Aishan Liu, Xianglong Liu

**Abstract**: Vision-language models (VLMs) have significantly advanced autonomous driving (AD) by enhancing reasoning capabilities. However, these models remain highly vulnerable to adversarial attacks. While existing research has primarily focused on general VLM attacks, the development of attacks tailored to the safety-critical AD context has been largely overlooked. In this paper, we take the first step toward designing adversarial attacks specifically targeting VLMs in AD, exposing the substantial risks these attacks pose within this critical domain. We identify two unique challenges for effective adversarial attacks on AD VLMs: the variability of textual instructions and the time-series nature of visual scenarios. To this end, we propose ADvLM, the first visual adversarial attack framework specifically designed for VLMs in AD. Our framework introduces Semantic-Invariant Induction, which uses a large language model to create a diverse prompt library of textual instructions with consistent semantic content, guided by semantic entropy. Building on this, we introduce Scenario-Associated Enhancement, an approach where attention mechanisms select key frames and perspectives within driving scenarios to optimize adversarial perturbations that generalize across the entire scenario. Extensive experiments on several AD VLMs over multiple benchmarks show that ADvLM achieves state-of-the-art attack effectiveness. Moreover, real-world attack studies further validate its applicability and potential in practice.



## **29. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

cs.MA

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.11782v2) [paper-pdf](http://arxiv.org/pdf/2410.11782v2)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.



## **30. Transferable Ensemble Black-box Jailbreak Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.23558v2) [paper-pdf](http://arxiv.org/pdf/2410.23558v2)

**Authors**: Yiqi Yang, Hongye Fu

**Abstract**: In this report, we propose a novel black-box jailbreak attacking framework that incorporates various LLM-as-Attacker methods to deliver transferable and powerful jailbreak attacks. Our method is designed based on three key observations from existing jailbreaking studies and practices. First, we consider an ensemble approach should be more effective in exposing the vulnerabilities of an aligned LLM compared to individual attacks. Second, different malicious instructions inherently vary in their jailbreaking difficulty, necessitating differentiated treatment to ensure more efficient attacks. Finally, the semantic coherence of a malicious instruction is crucial for triggering the defenses of an aligned LLM; therefore, it must be carefully disrupted to manipulate its embedding representation, thereby increasing the jailbreak success rate. We validated our approach by participating in the Competition for LLM and Agent Safety 2024, where our team achieved top performance in the Jailbreaking Attack Track.



## **31. R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edge**

eess.SP

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18220v1) [paper-pdf](http://arxiv.org/pdf/2411.18220v1)

**Authors**: Aladin Djuhera, Vlad C. Andrei, Mohsen Pourghasemian, Haris Gacanin, Holger Boche, Walid Saad

**Abstract**: Multi-task large language models (MTLLMs) are important for many applications at the wireless edge, where users demand specialized models to handle multiple tasks efficiently. However, training MTLLMs is complex and exhaustive, particularly when tasks are subject to change. Recently, the concept of model fusion via task vectors has emerged as an efficient approach for combining fine-tuning parameters to produce an MTLLM. In this paper, the problem of enabling edge users to collaboratively craft such MTTLMs via tasks vectors is studied, under the assumption of worst-case adversarial attacks. To this end, first the influence of adversarial noise to multi-task model fusion is investigated and a relationship between the so-called weight disentanglement error and the mean squared error (MSE) is derived. Using hypothesis testing, it is directly shown that the MSE increases interference between task vectors, thereby rendering model fusion ineffective. Then, a novel resilient MTLLM fusion (R-MTLLMF) is proposed, which leverages insights about the LLM architecture and fine-tuning process to safeguard task vector aggregation under adversarial noise by realigning the MTLLM. The proposed R-MTLLMF is then compared for both worst-case and ideal transmission scenarios to study the impact of the wireless channel. Extensive model fusion experiments with vision LLMs demonstrate R-MTLLMF's effectiveness, achieving close-to-baseline performance across eight different tasks in ideal noise scenarios and significantly outperforming unprotected model fusion in worst-case scenarios. The results further advocate for additional physical layer protection for a holistic approach to resilience, from both a wireless and LLM perspective.



## **32. Evaluating and Improving the Robustness of Security Attack Detectors Generated by LLMs**

cs.SE

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18216v1) [paper-pdf](http://arxiv.org/pdf/2411.18216v1)

**Authors**: Samuele Pasini, Jinhan Kim, Tommaso Aiello, Rocio Cabrera Lozoya, Antonino Sabetta, Paolo Tonella

**Abstract**: Large Language Models (LLMs) are increasingly used in software development to generate functions, such as attack detectors, that implement security requirements. However, LLMs struggle to generate accurate code, resulting, e.g., in attack detectors that miss well-known attacks when used in practice. This is most likely due to the LLM lacking knowledge about some existing attacks and to the generated code being not evaluated in real usage scenarios. We propose a novel approach integrating Retrieval Augmented Generation (RAG) and Self-Ranking into the LLM pipeline. RAG enhances the robustness of the output by incorporating external knowledge sources, while the Self-Ranking technique, inspired to the concept of Self-Consistency, generates multiple reasoning paths and creates ranks to select the most robust detector. Our extensive empirical study targets code generated by LLMs to detect two prevalent injection attacks in web security: Cross-Site Scripting (XSS) and SQL injection (SQLi). Results show a significant improvement in detection performance compared to baselines, with an increase of up to 71%pt and 37%pt in the F2-Score for XSS and SQLi detection, respectively.



## **33. Playing Language Game with LLMs Leads to Jailbreaking**

cs.CL

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.12762v2) [paper-pdf](http://arxiv.org/pdf/2411.12762v2)

**Authors**: Yu Peng, Zewen Long, Fangming Dong, Congyi Li, Shu Wu, Kai Chen

**Abstract**: The advent of large language models (LLMs) has spurred the development of numerous jailbreak techniques aimed at circumventing their security defenses against malicious attacks. An effective jailbreak approach is to identify a domain where safety generalization fails, a phenomenon known as mismatched generalization. In this paper, we introduce two novel jailbreak methods based on mismatched generalization: natural language games and custom language games, both of which effectively bypass the safety mechanisms of LLMs, with various kinds and different variants, making them hard to defend and leading to high attack rates. Natural language games involve the use of synthetic linguistic constructs and the actions intertwined with these constructs, such as the Ubbi Dubbi language. Building on this phenomenon, we propose the custom language games method: by engaging with LLMs using a variety of custom rules, we successfully execute jailbreak attacks across multiple LLM platforms. Extensive experiments demonstrate the effectiveness of our methods, achieving success rates of 93% on GPT-4o, 89% on GPT-4o-mini and 83% on Claude-3.5-Sonnet. Furthermore, to investigate the generalizability of safety alignments, we fine-tuned Llama-3.1-70B with the custom language games to achieve safety alignment within our datasets and found that when interacting through other language games, the fine-tuned models still failed to identify harmful content. This finding indicates that the safety alignment knowledge embedded in LLMs fails to generalize across different linguistic formats, thus opening new avenues for future research in this area.



## **34. BlackDAN: A Black-Box Multi-Objective Approach for Effective and Contextual Jailbreaking of Large Language Models**

cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.09804v3) [paper-pdf](http://arxiv.org/pdf/2410.09804v3)

**Authors**: Xinyuan Wang, Victor Shea-Jay Huang, Renmiao Chen, Hao Wang, Chengwei Pan, Lei Sha, Minlie Huang

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across various tasks, they encounter potential security risks such as jailbreak attacks, which exploit vulnerabilities to bypass security measures and generate harmful outputs. Existing jailbreak strategies mainly focus on maximizing attack success rate (ASR), frequently neglecting other critical factors, including the relevance of the jailbreak response to the query and the level of stealthiness. This narrow focus on single objectives can result in ineffective attacks that either lack contextual relevance or are easily recognizable. In this work, we introduce BlackDAN, an innovative black-box attack framework with multi-objective optimization, aiming to generate high-quality prompts that effectively facilitate jailbreaking while maintaining contextual relevance and minimizing detectability. BlackDAN leverages Multiobjective Evolutionary Algorithms (MOEAs), specifically the NSGA-II algorithm, to optimize jailbreaks across multiple objectives including ASR, stealthiness, and semantic relevance. By integrating mechanisms like mutation, crossover, and Pareto-dominance, BlackDAN provides a transparent and interpretable process for generating jailbreaks. Furthermore, the framework allows customization based on user preferences, enabling the selection of prompts that balance harmfulness, relevance, and other factors. Experimental results demonstrate that BlackDAN outperforms traditional single-objective methods, yielding higher success rates and improved robustness across various LLMs and multimodal LLMs, while ensuring jailbreak responses are both relevant and less detectable.



## **35. Desert Camels and Oil Sheikhs: Arab-Centric Red Teaming of Frontier LLMs**

cs.CL

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2410.24049v3) [paper-pdf](http://arxiv.org/pdf/2410.24049v3)

**Authors**: Muhammed Saeed, Elgizouli Mohamed, Mukhtar Mohamed, Shaina Raza, Muhammad Abdul-Mageed, Shady Shehata

**Abstract**: Large language models (LLMs) are widely used but raise ethical concerns due to embedded social biases. This study examines LLM biases against Arabs versus Westerners across eight domains, including women's rights, terrorism, and anti-Semitism and assesses model resistance to perpetuating these biases. To this end, we create two datasets: one to evaluate LLM bias toward Arabs versus Westerners and another to test model safety against prompts that exaggerate negative traits ("jailbreaks"). We evaluate six LLMs -- GPT-4, GPT-4o, LlaMA 3.1 (8B & 405B), Mistral 7B, and Claude 3.5 Sonnet. We find 79% of cases displaying negative biases toward Arabs, with LlaMA 3.1-405B being the most biased. Our jailbreak tests reveal GPT-4o as the most vulnerable, despite being an optimized version, followed by LlaMA 3.1-8B and Mistral 7B. All LLMs except Claude exhibit attack success rates above 87% in three categories. We also find Claude 3.5 Sonnet the safest, but it still displays biases in seven of eight categories. Despite being an optimized version of GPT4, We find GPT-4o to be more prone to biases and jailbreaks, suggesting optimization flaws. Our findings underscore the pressing need for more robust bias mitigation strategies and strengthened security measures in LLMs.



## **36. RTL-Breaker: Assessing the Security of LLMs against Backdoor Attacks on HDL Code Generation**

cs.CR

Accepted at 2025 Design, Automation & Test in Europe (DATE)  Conference

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17569v1) [paper-pdf](http://arxiv.org/pdf/2411.17569v1)

**Authors**: Lakshmi Likhitha Mankali, Jitendra Bhandari, Manaar Alam, Ramesh Karri, Michail Maniatakos, Ozgur Sinanoglu, Johann Knechtel

**Abstract**: Large language models (LLMs) have demonstrated remarkable potential with code generation/completion tasks for hardware design. In fact, LLM-based hardware description language (HDL) code generation has enabled the industry to realize complex designs more quickly, reducing the time and effort required in the development cycle. However, the increased reliance on such automation introduces critical security risks. Notably, given that LLMs have to be trained on vast datasets of codes that are typically sourced from publicly available repositories (often without thorough validation), LLMs are susceptible to so-called data poisoning or backdoor attacks. Here, attackers inject malicious code for the training data, which can be carried over into the HDL code generated by LLMs. This threat vector can compromise the security and integrity of entire hardware systems. In this work, we propose RTL-Breaker, a novel backdoor attack framework on LLM-based HDL code generation. RTL-Breaker provides an in-depth analysis for essential aspects of this novel problem: 1) various trigger mechanisms versus their effectiveness for inserting malicious modifications, and 2) side-effects by backdoor attacks on code generation in general, i.e., impact on code quality. RTL-Breaker emphasizes the urgent need for more robust measures to safeguard against such attacks. Toward that end, we open-source our framework and all data.



## **37. PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**

cs.CR

20 pages, 8 figures

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17453v1) [paper-pdf](http://arxiv.org/pdf/2411.17453v1)

**Authors**: Zhen Sun, Tianshuo Cong, Yule Liu, Chenhao Lin, Xinlei He, Rongmao Chen, Xingshuo Han, Xinyi Huang

**Abstract**: Fine-tuning is an essential process to improve the performance of Large Language Models (LLMs) in specific domains, with Parameter-Efficient Fine-Tuning (PEFT) gaining popularity due to its capacity to reduce computational demands through the integration of low-rank adapters. These lightweight adapters, such as LoRA, can be shared and utilized on open-source platforms. However, adversaries could exploit this mechanism to inject backdoors into these adapters, resulting in malicious behaviors like incorrect or harmful outputs, which pose serious security risks to the community. Unfortunately, few of the current efforts concentrate on analyzing the backdoor patterns or detecting the backdoors in the adapters.   To fill this gap, we first construct (and will release) PADBench, a comprehensive benchmark that contains 13,300 benign and backdoored adapters fine-tuned with various datasets, attack strategies, PEFT methods, and LLMs. Moreover, we propose PEFTGuard, the first backdoor detection framework against PEFT-based adapters. Extensive evaluation upon PADBench shows that PEFTGuard outperforms existing detection methods, achieving nearly perfect detection accuracy (100%) in most cases. Notably, PEFTGuard exhibits zero-shot transferability on three aspects, including different attacks, PEFT methods, and adapter ranks. In addition, we consider various adaptive attacks to demonstrate the high robustness of PEFTGuard. We further explore several possible backdoor mitigation defenses, finding fine-mixing to be the most effective method. We envision our benchmark and method can shed light on future LLM backdoor detection research.



## **38. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

cs.CL

Repo: https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2311.06062v4) [paper-pdf](http://arxiv.org/pdf/2311.06062v4)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Existing MIAs designed for large language models (LLMs) can be bifurcated into two types: reference-free and reference-based attacks. Although reference-based attacks appear promising performance by calibrating the probability measured on the target model with reference models, this illusion of privacy risk heavily depends on a reference dataset that closely resembles the training set. Both two types of attacks are predicated on the hypothesis that training records consistently maintain a higher probability of being sampled. However, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. Thus, these reasons lead to high false-positive rates of MIAs in practical scenarios. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs. Furthermore, we introduce probabilistic variation, a more reliable membership signal based on LLM memorization rather than overfitting, from which we rediscover the neighbour attack with theoretical grounding. Comprehensive evaluation conducted on three datasets and four exemplary LLMs shows that SPV-MIA raises the AUC of MIAs from 0.7 to a significantly high level of 0.9. Our code and dataset are available at: https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA



## **39. FATH: Authentication-based Test-time Defense against Indirect Prompt Injection Attacks**

cs.CR

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2410.21492v2) [paper-pdf](http://arxiv.org/pdf/2410.21492v2)

**Authors**: Jiongxiao Wang, Fangzhou Wu, Wendi Li, Jinsheng Pan, Edward Suh, Z. Morley Mao, Muhao Chen, Chaowei Xiao

**Abstract**: Large language models (LLMs) have been widely deployed as the backbone with additional tools and text information for real-world applications. However, integrating external information into LLM-integrated applications raises significant security concerns. Among these, prompt injection attacks are particularly threatening, where malicious instructions injected in the external text information can exploit LLMs to generate answers as the attackers desire. While both training-time and test-time defense methods have been developed to mitigate such attacks, the unaffordable training costs associated with training-time methods and the limited effectiveness of existing test-time methods make them impractical. This paper introduces a novel test-time defense strategy, named Formatting AuThentication with Hash-based tags (FATH). Unlike existing approaches that prevent LLMs from answering additional instructions in external text, our method implements an authentication system, requiring LLMs to answer all received instructions with a security policy and selectively filter out responses to user instructions as the final output. To achieve this, we utilize hash-based authentication tags to label each response, facilitating accurate identification of responses according to the user's instructions and improving the robustness against adaptive attacks. Comprehensive experiments demonstrate that our defense method can effectively defend against indirect prompt injection attacks, achieving state-of-the-art performance under Llama3 and GPT3.5 models across various attack methods. Our code is released at: https://github.com/Jayfeather1024/FATH



## **40. Just-in-Time Detection of Silent Security Patches**

cs.CR

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2312.01241v3) [paper-pdf](http://arxiv.org/pdf/2312.01241v3)

**Authors**: Xunzhu Tang, Zhenghan Chen, Kisub Kim, Haoye Tian, Saad Ezzini, Jacques Klein

**Abstract**: Open-source code is pervasive. In this setting, embedded vulnerabilities are spreading to downstream software at an alarming rate. While such vulnerabilities are generally identified and addressed rapidly, inconsistent maintenance policies may lead security patches to go unnoticed. Indeed, security patches can be {\em silent}, i.e., they do not always come with comprehensive advisories such as CVEs. This lack of transparency leaves users oblivious to available security updates, providing ample opportunity for attackers to exploit unpatched vulnerabilities. Consequently, identifying silent security patches just in time when they are released is essential for preventing n-day attacks, and for ensuring robust and secure maintenance practices. With LLMDA we propose to (1) leverage large language models (LLMs) to augment patch information with generated code change explanations, (2) design a representation learning approach that explores code-text alignment methodologies for feature combination, (3) implement a label-wise training with labelled instructions for guiding the embedding based on security relevance, and (4) rely on a probabilistic batch contrastive learning mechanism for building a high-precision identifier of security patches. We evaluate LLMDA on the PatchDB and SPI-DB literature datasets and show that our approach substantially improves over the state-of-the-art, notably GraphSPD by 20% in terms of F-Measure on the SPI-DB benchmark.



## **41. Scaling Laws for Black box Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16782v1) [paper-pdf](http://arxiv.org/pdf/2411.16782v1)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: A longstanding problem of deep learning models is their vulnerability to adversarial examples, which are often generated by applying imperceptible perturbations to natural examples. Adversarial examples exhibit cross-model transferability, enabling to attack black-box models with limited information about their architectures and parameters. Model ensembling is an effective strategy to improve the transferability by attacking multiple surrogate models simultaneously. However, as prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the findings in large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. By analyzing the relationship between the number of surrogate models and transferability of adversarial examples, we conclude with clear scaling laws, emphasizing the potential of using more surrogate models to enhance adversarial transferability. Extensive experiments verify the claims on standard image classifiers, multimodal large language models, and even proprietary models like GPT-4o, demonstrating consistent scaling effects and impressive attack success rates with more surrogate models. Further studies by visualization indicate that scaled attacks bring better interpretability in semantics, indicating that the common features of models are captured.



## **42. LLMPirate: LLMs for Black-box Hardware IP Piracy**

cs.CR

Accepted by NDSS Symposium 2025

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16111v1) [paper-pdf](http://arxiv.org/pdf/2411.16111v1)

**Authors**: Vasudev Gohil, Matthew DeLorenzo, Veera Vishwa Achuta Sai Venkat Nallam, Joey See, Jeyavijayan Rajendran

**Abstract**: The rapid advancement of large language models (LLMs) has enabled the ability to effectively analyze and generate code nearly instantaneously, resulting in their widespread adoption in software development. Following this advancement, researchers and companies have begun integrating LLMs across the hardware design and verification process. However, these highly potent LLMs can also induce new attack scenarios upon security vulnerabilities across the hardware development process. One such attack vector that has not been explored is intellectual property (IP) piracy. Given that this attack can manifest as rewriting hardware designs to evade piracy detection, it is essential to thoroughly evaluate LLM capabilities in performing this task and assess the mitigation abilities of current IP piracy detection tools.   Therefore, in this work, we propose LLMPirate, the first LLM-based technique able to generate pirated variations of circuit designs that successfully evade detection across multiple state-of-the-art piracy detection tools. We devise three solutions to overcome challenges related to integration of LLMs for hardware circuit designs, scalability to large circuits, and effectiveness, resulting in an end-to-end automated, efficient, and practical formulation. We perform an extensive experimental evaluation of LLMPirate using eight LLMs of varying sizes and capabilities and assess their performance in pirating various circuit designs against four state-of-the-art, widely-used piracy detection tools. Our experiments demonstrate that LLMPirate is able to consistently evade detection on 100% of tested circuits across every detection tool. Additionally, we showcase the ramifications of LLMPirate using case studies on IBEX and MOR1KX processors and a GPS module, that we successfully pirate. We envision that our work motivates and fosters the development of better IP piracy detection tools.



## **43. In-Context Experience Replay Facilitates Safety Red-Teaming of Text-to-Image Diffusion Models**

cs.LG

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16769v1) [paper-pdf](http://arxiv.org/pdf/2411.16769v1)

**Authors**: Zhi-Yi Chin, Kuan-Chen Mu, Mario Fritz, Pin-Yu Chen, Wei-Chen Chiu

**Abstract**: Text-to-image (T2I) models have shown remarkable progress, but their potential to generate harmful content remains a critical concern in the ML community. While various safety mechanisms have been developed, the field lacks systematic tools for evaluating their effectiveness against real-world misuse scenarios. In this work, we propose ICER, a novel red-teaming framework that leverages Large Language Models (LLMs) and a bandit optimization-based algorithm to generate interpretable and semantic meaningful problematic prompts by learning from past successful red-teaming attempts. Our ICER efficiently probes safety mechanisms across different T2I models without requiring internal access or additional training, making it broadly applicable to deployed systems. Through extensive experiments, we demonstrate that ICER significantly outperforms existing prompt attack methods in identifying model vulnerabilities while maintaining high semantic similarity with intended content. By uncovering that successful jailbreaking instances can systematically facilitate the discovery of new vulnerabilities, our work provides crucial insights for developing more robust safety mechanisms in T2I systems.



## **44. Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning Attack**

cs.LG

Rejected by ICML2024. Accepted by NeurIPS2024

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2402.01109v6) [paper-pdf](http://arxiv.org/pdf/2402.01109v6)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.



## **45. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

cs.CL

Published as a conference paper at COLM 2024  (https://colmweb.org/index.html)

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2404.07921v3) [paper-pdf](http://arxiv.org/pdf/2404.07921v3)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.



## **46. JailBreakV: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2404.03027v4) [paper-pdf](http://arxiv.org/pdf/2404.03027v4)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.



## **47. InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models**

cs.CL

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2410.22770v2) [paper-pdf](http://arxiv.org/pdf/2410.22770v2)

**Authors**: Hao Li, Xiaogeng Liu

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), enabling goal hijacking and data leakage. Prompt guard models, though effective in defense, suffer from over-defense -- falsely flagging benign inputs as malicious due to trigger word bias. To address this issue, we introduce NotInject, an evaluation dataset that systematically measures over-defense across various prompt guard models. NotInject contains 339 benign samples enriched with trigger words common in prompt injection attacks, enabling fine-grained evaluation. Our results show that state-of-the-art models suffer from over-defense issues, with accuracy dropping close to random guessing levels (60%). To mitigate this, we propose InjecGuard, a novel prompt guard model that incorporates a new training strategy, Mitigating Over-defense for Free (MOF), which significantly reduces the bias on trigger words. InjecGuard demonstrates state-of-the-art performance on diverse benchmarks including NotInject, surpassing the existing best model by 30.8%, offering a robust and open-source solution for detecting prompt injection attacks. The code and datasets are released at https://github.com/SaFoLab-WISC/InjecGuard.



## **48. Semantic Shield: Defending Vision-Language Models Against Backdooring and Poisoning via Fine-grained Knowledge Alignment**

cs.CV

CVPR 2024

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2411.15673v1) [paper-pdf](http://arxiv.org/pdf/2411.15673v1)

**Authors**: Alvi Md Ishmam, Christopher Thomas

**Abstract**: In recent years there has been enormous interest in vision-language models trained using self-supervised objectives. However, the use of large-scale datasets scraped from the web for training also makes these models vulnerable to potential security threats, such as backdooring and poisoning attacks. In this paper, we propose a method for mitigating such attacks on contrastively trained vision-language models. Our approach leverages external knowledge extracted from a language model to prevent models from learning correlations between image regions which lack strong alignment with external knowledge. We do this by imposing constraints to enforce that attention paid by the model to visual regions is proportional to the alignment of those regions with external knowledge. We conduct extensive experiments using a variety of recent backdooring and poisoning attacks on multiple datasets and architectures. Our results clearly demonstrate that our proposed approach is highly effective at defending against such attacks across multiple settings, while maintaining model utility and without requiring any changes at inference time



## **49. Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning**

cs.LG

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.14937v1) [paper-pdf](http://arxiv.org/pdf/2411.14937v1)

**Authors**: Junjie Shan, Ziqi Zhao, Jialin Lu, Rui Zhang, Siu Ming Yiu, Ka-Ho Chow

**Abstract**: Foundation models that bridge vision and language have made significant progress, inspiring numerous life-enriching applications. However, their potential for misuse to introduce new threats remains largely unexplored. This paper reveals that vision-language models (VLMs) can be exploited to overcome longstanding limitations in gradient inversion attacks (GIAs) within federated learning (FL), where an FL server reconstructs private data samples from gradients shared by victim clients. Current GIAs face challenges in reconstructing high-resolution images, especially when the victim has a large local data batch. While focusing reconstruction on valuable samples rather than the entire batch is promising, existing methods lack the flexibility to allow attackers to specify their target data. In this paper, we introduce Geminio, the first approach to transform GIAs into semantically meaningful, targeted attacks. Geminio enables a brand new privacy attack experience: attackers can describe, in natural language, the types of data they consider valuable, and Geminio will prioritize reconstruction to focus on those high-value samples. This is achieved by leveraging a pretrained VLM to guide the optimization of a malicious global model that, when shared with and optimized by a victim, retains only gradients of samples that match the attacker-specified query. Extensive experiments demonstrate Geminio's effectiveness in pinpointing and reconstructing targeted samples, with high success rates across complex datasets under FL and large batch sizes and showing resilience against existing defenses.



## **50. Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Language Models**

cs.SD

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.14842v1) [paper-pdf](http://arxiv.org/pdf/2411.14842v1)

**Authors**: Wanqi Yang, Yanda Li, Meng Fang, Yunchao Wei, Tianyi Zhou, Ling Chen

**Abstract**: Adversarial audio attacks pose a significant threat to the growing use of large language models (LLMs) in voice-based human-machine interactions. While existing research has primarily focused on model-specific adversarial methods, real-world applications demand a more generalizable and universal approach to audio adversarial attacks. In this paper, we introduce the Chat-Audio Attacks (CAA) benchmark including four distinct types of audio attacks, which aims to explore the the vulnerabilities of LLMs to these audio attacks in conversational scenarios. To evaluate the robustness of LLMs, we propose three evaluation strategies: Standard Evaluation, utilizing traditional metrics to quantify model performance under attacks; GPT-4o-Based Evaluation, which simulates real-world conversational complexities; and Human Evaluation, offering insights into user perception and trust. We evaluate six state-of-the-art LLMs with voice interaction capabilities, including Gemini-1.5-Pro, GPT-4o, and others, using three distinct evaluation methods on the CAA benchmark. Our comprehensive analysis reveals the impact of four types of audio attacks on the performance of these models, demonstrating that GPT-4o exhibits the highest level of resilience.



