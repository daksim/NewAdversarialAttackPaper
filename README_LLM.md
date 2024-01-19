# Latest Large Language Model Attack Papers
**update at 2024-01-19 11:39:03**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks**

cs.CL

11 pages, 3 figures, 2 tables

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09798v1) [paper-pdf](http://arxiv.org/pdf/2401.09798v1)

**Authors**: Kazuhiro Takemoto

**Abstract**: Large Language Models (LLMs) like ChatGPT face `jailbreak' challenges, where safeguards are bypassed to produce ethically harmful prompts. This study introduces a simple black-box method to effectively generate jailbreak prompts, overcoming the limitations of high complexity and computational costs associated with existing methods. The proposed technique iteratively rewrites harmful prompts into non-harmful expressions using the target LLM itself, based on the hypothesis that LLMs can directly sample safeguard-bypassing expressions. Demonstrated through experiments with ChatGPT (GPT-3.5 and GPT-4) and Gemini-Pro, this method achieved an attack success rate of over 80% within an average of 5 iterations and remained effective despite model updates. The jailbreak prompts generated were naturally-worded and concise, suggesting they are less detectable. The results indicate that creating effective jailbreak prompts is simpler than previously considered, and black-box jailbreak attacks pose a more serious security threat.



## **2. Large Language Model Lateral Spear Phishing: A Comparative Study in Large-Scale Organizational Settings**

cs.CR

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09727v1) [paper-pdf](http://arxiv.org/pdf/2401.09727v1)

**Authors**: Mazal Bethany, Athanasios Galiopoulos, Emet Bethany, Mohammad Bahrami Karkevandi, Nishant Vishwamitra, Peyman Najafirad

**Abstract**: The critical threat of phishing emails has been further exacerbated by the potential of LLMs to generate highly targeted, personalized, and automated spear phishing attacks. Two critical problems concerning LLM-facilitated phishing require further investigation: 1) Existing studies on lateral phishing lack specific examination of LLM integration for large-scale attacks targeting the entire organization, and 2) Current anti-phishing infrastructure, despite its extensive development, lacks the capability to prevent LLM-generated attacks, potentially impacting both employees and IT security incident management. However, the execution of such investigative studies necessitates a real-world environment, one that functions during regular business operations and mirrors the complexity of a large organizational infrastructure. This setting must also offer the flexibility required to facilitate a diverse array of experimental conditions, particularly the incorporation of phishing emails crafted by LLMs. This study is a pioneering exploration into the use of Large Language Models (LLMs) for the creation of targeted lateral phishing emails, targeting a large tier 1 university's operation and workforce of approximately 9,000 individuals over an 11-month period. It also evaluates the capability of email filtering infrastructure to detect such LLM-generated phishing attempts, providing insights into their effectiveness and identifying potential areas for improvement. Based on our findings, we propose machine learning-based detection techniques for such emails to detect LLM-generated phishing emails that were missed by the existing infrastructure, with an F1-score of 98.96.



## **3. MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance**

cs.CR

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.02906v2) [paper-pdf](http://arxiv.org/pdf/2401.02906v2)

**Authors**: Renjie Pi, Tianyang Han, Yueqi Xie, Rui Pan, Qing Lian, Hanze Dong, Jipeng Zhang, Tong Zhang

**Abstract**: The deployment of multimodal large language models (MLLMs) has brought forth a unique vulnerability: susceptibility to malicious attacks through visual inputs. We delve into the novel challenge of defending MLLMs against such attacks. We discovered that images act as a "foreign language" that is not considered during alignment, which can make MLLMs prone to producing harmful responses. Unfortunately, unlike the discrete tokens considered in text-based LLMs, the continuous nature of image signals presents significant alignment challenges, which poses difficulty to thoroughly cover the possible scenarios. This vulnerability is exacerbated by the fact that open-source MLLMs are predominantly fine-tuned on limited image-text pairs that is much less than the extensive text-based pretraining corpus, which makes the MLLMs more prone to catastrophic forgetting of their original abilities during explicit alignment tuning. To tackle these challenges, we introduce MLLM-Protector, a plug-and-play strategy combining a lightweight harm detector and a response detoxifier. The harm detector's role is to identify potentially harmful outputs from the MLLM, while the detoxifier corrects these outputs to ensure the response stipulates to the safety standards. This approach effectively mitigates the risks posed by malicious visual inputs without compromising the model's overall performance. Our results demonstrate that MLLM-Protector offers a robust solution to a previously unaddressed aspect of MLLM security.



## **4. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

cs.CL

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09002v1) [paper-pdf](http://arxiv.org/pdf/2401.09002v1)

**Authors**: Dong shu, Mingyu Jin, Suiyuan Zhu, Beichen Wang, Zihao Zhou, Chong Zhang, Yongfeng Zhang

**Abstract**: In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation aligns with the baseline's trend while offering a more profound and detailed assessment. We believe that by accurately evaluating the effectiveness of attack prompts in the Jailbreak task, our work lays a solid foundation for assessing a wider array of similar or even more complex tasks in the realm of prompt injection, potentially revolutionizing this field.



## **5. Whispering Pixels: Exploiting Uninitialized Register Accesses in Modern GPUs**

cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08881v1) [paper-pdf](http://arxiv.org/pdf/2401.08881v1)

**Authors**: Frederik Dermot Pustelnik, Xhani Marvin Saß, Jean-Pierre Seifert

**Abstract**: Graphic Processing Units (GPUs) have transcended their traditional use-case of rendering graphics and nowadays also serve as a powerful platform for accelerating ubiquitous, non-graphical rendering tasks. One prominent task is inference of neural networks, which process vast amounts of personal data, such as audio, text or images. Thus, GPUs became integral components for handling vast amounts of potentially confidential data, which has awakened the interest of security researchers. This lead to the discovery of various vulnerabilities in GPUs in recent years. In this paper, we uncover yet another vulnerability class in GPUs: We found that some GPU implementations lack proper register initialization routines before shader execution, leading to unintended register content leakage of previously executed shader kernels. We showcase the existence of the aforementioned vulnerability on products of 3 major vendors - Apple, NVIDIA and Qualcomm. The vulnerability poses unique challenges to an adversary due to opaque scheduling and register remapping algorithms present in the GPU firmware, complicating the reconstruction of leaked data. In order to illustrate the real-world impact of this flaw, we showcase how these challenges can be solved for attacking various workloads on the GPU. First, we showcase how uninitialized registers leak arbitrary pixel data processed by fragment shaders. We further implement information leakage attacks on intermediate data of Convolutional Neural Networks (CNNs) and present the attack's capability to leak and reconstruct the output of Large Language Models (LLMs).



## **6. IsamasRed: A Public Dataset Tracking Reddit Discussions on Israel-Hamas Conflict**

cs.SI

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08202v1) [paper-pdf](http://arxiv.org/pdf/2401.08202v1)

**Authors**: Kai Chen, Zihao He, Keith Burghardt, Jingxin Zhang, Kristina Lerman

**Abstract**: The conflict between Israel and Palestinians significantly escalated after the October 7, 2023 Hamas attack, capturing global attention. To understand the public discourse on this conflict, we present a meticulously compiled dataset--IsamasRed--comprising nearly 400,000 conversations and over 8 million comments from Reddit, spanning from August 2023 to November 2023. We introduce an innovative keyword extraction framework leveraging a large language model to effectively identify pertinent keywords, ensuring a comprehensive data collection. Our initial analysis on the dataset, examining topics, controversy, emotional and moral language trends over time, highlights the emotionally charged and complex nature of the discourse. This dataset aims to enrich the understanding of online discussions, shedding light on the complex interplay between ideology, sentiment, and community engagement in digital spaces.



## **7. MGTBench: Benchmarking Machine-Generated Text Detection**

cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2303.14822v3) [paper-pdf](http://arxiv.org/pdf/2303.14822v3)

**Authors**: Xinlei He, Xinyue Shen, Zeyuan Chen, Michael Backes, Yang Zhang

**Abstract**: Nowadays, powerful large language models (LLMs) such as ChatGPT have demonstrated revolutionary power in a variety of tasks. Consequently, the detection of machine-generated texts (MGTs) is becoming increasingly crucial as LLMs become more advanced and prevalent. These models have the ability to generate human-like language, making it challenging to discern whether a text is authored by a human or a machine. This raises concerns regarding authenticity, accountability, and potential bias. However, existing methods for detecting MGTs are evaluated using different model architectures, datasets, and experimental settings, resulting in a lack of a comprehensive evaluation framework that encompasses various methodologies. Furthermore, it remains unclear how existing detection methods would perform against powerful LLMs. In this paper, we fill this gap by proposing the first benchmark framework for MGT detection against powerful LLMs, named MGTBench. Extensive evaluations on public datasets with curated texts generated by various powerful LLMs such as ChatGPT-turbo and Claude demonstrate the effectiveness of different detection methods. Our ablation study shows that a larger number of words in general leads to better performance and most detection methods can achieve similar performance with much fewer training samples. Moreover, we delve into a more challenging task: text attribution. Our findings indicate that the model-based detection methods still perform well in the text attribution task. To investigate the robustness of different detection methods, we consider three adversarial attacks, namely paraphrasing, random spacing, and adversarial perturbations. We discover that these attacks can significantly diminish detection effectiveness, underscoring the critical need for the development of more robust detection methods.



## **8. Traces of Memorisation in Large Language Models for Code**

cs.CR

ICSE 2024 Research Track

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2312.11658v2) [paper-pdf](http://arxiv.org/pdf/2312.11658v2)

**Authors**: Ali Al-Kaswan, Maliheh Izadi, Arie van Deursen

**Abstract**: Large language models have gained significant popularity because of their ability to generate human-like text and potential applications in various fields, such as Software Engineering. Large language models for code are commonly trained on large unsanitised corpora of source code scraped from the internet. The content of these datasets is memorised and can be extracted by attackers with data extraction attacks. In this work, we explore memorisation in large language models for code and compare the rate of memorisation with large language models trained on natural language. We adopt an existing benchmark for natural language and construct a benchmark for code by identifying samples that are vulnerable to attack. We run both benchmarks against a variety of models, and perform a data extraction attack. We find that large language models for code are vulnerable to data extraction attacks, like their natural language counterparts. From the training data that was identified to be potentially extractable we were able to extract 47% from a CodeGen-Mono-16B code completion model. We also observe that models memorise more, as their parameter count grows, and that their pre-training data are also vulnerable to attack. We also find that data carriers are memorised at a higher rate than regular code or documentation and that different model architectures memorise different samples. Data leakage has severe outcomes, so we urge the research community to further investigate the extent of this phenomenon using a wider range of models and extraction techniques in order to build safeguards to mitigate this issue.



## **9. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

cs.CL

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07867v1) [paper-pdf](http://arxiv.org/pdf/2401.07867v1)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of latest Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause detection evasion in all tested languages, where homoglyph attacks are especially successful.



## **10. Signed-Prompt: A New Approach to Prevent Prompt Injection Attacks Against LLM-Integrated Applications**

cs.CR

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07612v1) [paper-pdf](http://arxiv.org/pdf/2401.07612v1)

**Authors**: Xuchen Suo

**Abstract**: The critical challenge of prompt injection attacks in Large Language Models (LLMs) integrated applications, a growing concern in the Artificial Intelligence (AI) field. Such attacks, which manipulate LLMs through natural language inputs, pose a significant threat to the security of these applications. Traditional defense strategies, including output and input filtering, as well as delimiter use, have proven inadequate. This paper introduces the 'Signed-Prompt' method as a novel solution. The study involves signing sensitive instructions within command segments by authorized users, enabling the LLM to discern trusted instruction sources. The paper presents a comprehensive analysis of prompt injection attack patterns, followed by a detailed explanation of the Signed-Prompt concept, including its basic architecture and implementation through both prompt engineering and fine-tuning of LLMs. Experiments demonstrate the effectiveness of the Signed-Prompt method, showing substantial resistance to various types of prompt injection attacks, thus validating its potential as a robust defense strategy in AI security.



## **11. Stability Analysis of ChatGPT-based Sentiment Analysis in AI Quality Assurance**

cs.CL

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07441v1) [paper-pdf](http://arxiv.org/pdf/2401.07441v1)

**Authors**: Tinghui Ouyang, AprilPyone MaungMaung, Koichi Konishi, Yoshiki Seo, Isao Echizen

**Abstract**: In the era of large AI models, the complex architecture and vast parameters present substantial challenges for effective AI quality management (AIQM), e.g. large language model (LLM). This paper focuses on investigating the quality assurance of a specific LLM-based AI product--a ChatGPT-based sentiment analysis system. The study delves into stability issues related to both the operation and robustness of the expansive AI model on which ChatGPT is based. Experimental analysis is conducted using benchmark datasets for sentiment analysis. The results reveal that the constructed ChatGPT-based sentiment analysis system exhibits uncertainty, which is attributed to various operational factors. It demonstrated that the system also exhibits stability issues in handling conventional small text attacks involving robustness.



## **12. Advancing TTP Analysis: Harnessing the Power of Encoder-Only and Decoder-Only Language Models with Retrieval Augmented Generation**

cs.CR

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.00280v2) [paper-pdf](http://arxiv.org/pdf/2401.00280v2)

**Authors**: Reza Fayyazi, Rozhina Taghdimi, Shanchieh Jay Yang

**Abstract**: Tactics, Techniques, and Procedures (TTPs) outline the methods attackers use to exploit vulnerabilities. The interpretation of TTPs in the MITRE ATT&CK framework can be challenging for cybersecurity practitioners due to presumed expertise, complex dependencies, and inherent ambiguity. Meanwhile, advancements with Large Language Models (LLMs) have led to recent surge in studies exploring its uses in cybersecurity operations. This leads us to question how well encoder-only (e.g., RoBERTa) and decoder-only (e.g., GPT-3.5) LLMs can comprehend and summarize TTPs to inform analysts of the intended purposes (i.e., tactics) of a cyberattack procedure. The state-of-the-art LLMs have shown to be prone to hallucination by providing inaccurate information, which is problematic in critical domains like cybersecurity. Therefore, we propose the use of Retrieval Augmented Generation (RAG) techniques to extract relevant contexts for each cyberattack procedure for decoder-only LLMs (without fine-tuning). We further contrast such approach against supervised fine-tuning (SFT) of encoder-only LLMs. Our results reveal that both the direct-use of decoder-only LLMs (i.e., its pre-trained knowledge) and the SFT of encoder-only LLMs offer inaccurate interpretation of cyberattack procedures. Significant improvements are shown when RAG is used for decoder-only LLMs, particularly when directly relevant context is found. This study further sheds insights on the limitations and capabilities of using RAG for LLMs in interpreting TTPs.



## **13. How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs**

cs.CL

14 pages of the main text, qualitative examples of jailbreaks may be  harmful in nature

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.06373v1) [paper-pdf](http://arxiv.org/pdf/2401.06373v1)

**Authors**: Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi

**Abstract**: Most traditional AI safety research has approached AI models as machines and centered on algorithm-focused attacks developed by security experts. As large language models (LLMs) become increasingly common and competent, non-expert users can also impose risks during daily interactions. This paper introduces a new perspective to jailbreak LLMs as human-like communicators, to explore this overlooked intersection between everyday language interaction and AI safety. Specifically, we study how to persuade LLMs to jailbreak them. First, we propose a persuasion taxonomy derived from decades of social science research. Then, we apply the taxonomy to automatically generate interpretable persuasive adversarial prompts (PAP) to jailbreak LLMs. Results show that persuasion significantly increases the jailbreak performance across all risk categories: PAP consistently achieves an attack success rate of over $92\%$ on Llama 2-7b Chat, GPT-3.5, and GPT-4 in $10$ trials, surpassing recent algorithm-focused attacks. On the defense side, we explore various mechanisms against PAP and, found a significant gap in existing defenses, and advocate for more fundamental mitigation for highly interactive LLMs



## **14. Intention Analysis Prompting Makes Large Language Models A Good Jailbreak Defender**

cs.CL

9 pages, 5 figures

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.06561v1) [paper-pdf](http://arxiv.org/pdf/2401.06561v1)

**Authors**: Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao

**Abstract**: Aligning large language models (LLMs) with human values, particularly in the face of stealthy and complex jailbreaks, presents a formidable challenge. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis Prompting (IAPrompt). The principle behind is to trigger LLMs' inherent self-correct and improve ability through a two-stage process: 1) essential intention analysis, and 2) policy-aligned response. Notably, IAPrompt is an inference-only method, thus could enhance the safety of LLMs without compromising their helpfulness. Extensive experiments on SAP200 and DAN benchmarks across Vicuna, ChatGLM, MPT, DeepSeek, and GPT-3.5 show that IAPrompt could consistently and significantly reduce the harmfulness in response (averagely -46.5% attack success rate) and maintain the general helpfulness. Further analyses present some insights into how our method works. To facilitate reproducibility, We release our code and scripts at: https://github.com/alphadl/SafeLLM_with_IntentionAnalysis



## **15. Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks**

cs.CL

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.05949v2) [paper-pdf](http://arxiv.org/pdf/2401.05949v2)

**Authors**: Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Jinming Wen

**Abstract**: In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Unlike traditional fine-tuning methods, in-context learning adapts pre-trained models to unseen tasks without updating any parameters. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we have designed a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning prompts, which can make models behave in accordance with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 40B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models. Our findings highlight the vulnerabilities of language models, and we hope this work will raise awareness of the possible security threats associated with in-context learning.



## **16. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

cs.CL

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2310.13191v3) [paper-pdf](http://arxiv.org/pdf/2310.13191v3)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.



## **17. LimeAttack: Local Explainable Method for Textual Hard-Label Adversarial Attack**

cs.CL

18 pages, 38th AAAI Main Track

**SubmitDate**: 2024-01-10    [abs](http://arxiv.org/abs/2308.00319v2) [paper-pdf](http://arxiv.org/pdf/2308.00319v2)

**Authors**: Hai Zhu, Zhaoqing Yang, Weiwei Shang, Yuren Wu

**Abstract**: Natural language processing models are vulnerable to adversarial examples. Previous textual adversarial attacks adopt gradients or confidence scores to calculate word importance ranking and generate adversarial examples. However, this information is unavailable in the real world. Therefore, we focus on a more realistic and challenging setting, named hard-label attack, in which the attacker can only query the model and obtain a discrete prediction label. Existing hard-label attack algorithms tend to initialize adversarial examples by random substitution and then utilize complex heuristic algorithms to optimize the adversarial perturbation. These methods require a lot of model queries and the attack success rate is restricted by adversary initialization. In this paper, we propose a novel hard-label attack algorithm named LimeAttack, which leverages a local explainable method to approximate word importance ranking, and then adopts beam search to find the optimal solution. Extensive experiments show that LimeAttack achieves the better attacking performance compared with existing hard-label attack under the same query budget. In addition, we evaluate the effectiveness of LimeAttack on large language models, and results indicate that adversarial examples remain a significant threat to large language models. The adversarial examples crafted by LimeAttack are highly transferable and effectively improve model robustness in adversarial training.



## **18. Jatmo: Prompt Injection Defense by Task-Specific Finetuning**

cs.CR

24 pages, 6 figures

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2312.17673v2) [paper-pdf](http://arxiv.org/pdf/2312.17673v2)

**Authors**: Julien Piet, Maha Alrashed, Chawin Sitawarin, Sizhe Chen, Zeming Wei, Elizabeth Sun, Basel Alomair, David Wagner

**Abstract**: Large Language Models (LLMs) are attracting significant research attention due to their instruction-following abilities, allowing users and developers to leverage LLMs for a variety of tasks. However, LLMs are vulnerable to prompt-injection attacks: a class of attacks that hijack the model's instruction-following abilities, changing responses to prompts to undesired, possibly malicious ones. In this work, we introduce Jatmo, a method for generating task-specific models resilient to prompt-injection attacks. Jatmo leverages the fact that LLMs can only follow instructions once they have undergone instruction tuning. It harnesses a teacher instruction-tuned model to generate a task-specific dataset, which is then used to fine-tune a base model (i.e., a non-instruction-tuned model). Jatmo only needs a task prompt and a dataset of inputs for the task: it uses the teacher model to generate outputs. For situations with no pre-existing datasets, Jatmo can use a single example, or in some cases none at all, to produce a fully synthetic dataset. Our experiments on seven tasks show that Jatmo models provide similar quality of outputs on their specific task as standard LLMs, while being resilient to prompt injections. The best attacks succeeded in less than 0.5% of cases against our models, versus 87% success rate against GPT-3.5-Turbo. We release Jatmo at https://github.com/wagner-group/prompt-injection-defense.



## **19. The Stronger the Diffusion Model, the Easier the Backdoor: Data Poisoning to Induce Copyright Breaches Without Adjusting Finetuning Pipeline**

cs.CR

This study reveals that by subtly inserting non-copyright-infringing  poisoning data into a diffusion model's training dataset, it's possible to  trigger the model to generate copyrighted content, highlighting  vulnerabilities in current copyright protection strategies

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2401.04136v1) [paper-pdf](http://arxiv.org/pdf/2401.04136v1)

**Authors**: Haonan Wang, Qianli Shen, Yao Tong, Yang Zhang, Kenji Kawaguchi

**Abstract**: The commercialization of diffusion models, renowned for their ability to generate high-quality images that are often indistinguishable from real ones, brings forth potential copyright concerns. Although attempts have been made to impede unauthorized access to copyrighted material during training and to subsequently prevent DMs from generating copyrighted images, the effectiveness of these solutions remains unverified. This study explores the vulnerabilities associated with copyright protection in DMs by introducing a backdoor data poisoning attack (SilentBadDiffusion) against text-to-image diffusion models. Our attack method operates without requiring access to or control over the diffusion model's training or fine-tuning processes; it merely involves the insertion of poisoning data into the clean training dataset. This data, comprising poisoning images equipped with prompts, is generated by leveraging the powerful capabilities of multimodal large language models and text-guided image inpainting techniques. Our experimental results and analysis confirm the method's effectiveness. By integrating a minor portion of non-copyright-infringing stealthy poisoning data into the clean dataset-rendering it free from suspicion-we can prompt the finetuned diffusion models to produce copyrighted content when activated by specific trigger prompts. These findings underline potential pitfalls in the prevailing copyright protection strategies and underscore the necessity for increased scrutiny and preventative measures against the misuse of DMs.



## **20. PromptBench: A Unified Library for Evaluation of Large Language Models**

cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2312.07910v2) [paper-pdf](http://arxiv.org/pdf/2312.07910v2)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.



## **21. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

cs.CV

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.01886v2) [paper-pdf](http://arxiv.org/pdf/2312.01886v2)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.



## **22. Mining Temporal Attack Patterns from Cyberthreat Intelligence Reports**

cs.CR

A modified version of this pre-print is submitted to IEEE  Transactions on Software Engineering, and is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01883v1) [paper-pdf](http://arxiv.org/pdf/2401.01883v1)

**Authors**: Md Rayhanur Rahman, Brandon Wroblewski, Quinn Matthews, Brantley Morgan, Tim Menzies, Laurie Williams

**Abstract**: Defending from cyberattacks requires practitioners to operate on high-level adversary behavior. Cyberthreat intelligence (CTI) reports on past cyberattack incidents describe the chain of malicious actions with respect to time. To avoid repeating cyberattack incidents, practitioners must proactively identify and defend against recurring chain of actions - which we refer to as temporal attack patterns. Automatically mining the patterns among actions provides structured and actionable information on the adversary behavior of past cyberattacks. The goal of this paper is to aid security practitioners in prioritizing and proactive defense against cyberattacks by mining temporal attack patterns from cyberthreat intelligence reports. To this end, we propose ChronoCTI, an automated pipeline for mining temporal attack patterns from cyberthreat intelligence (CTI) reports of past cyberattacks. To construct ChronoCTI, we build the ground truth dataset of temporal attack patterns and apply state-of-the-art large language models, natural language processing, and machine learning techniques. We apply ChronoCTI on a set of 713 CTI reports, where we identify 124 temporal attack patterns - which we categorize into nine pattern categories. We identify that the most prevalent pattern category is to trick victim users into executing malicious code to initiate the attack, followed by bypassing the anti-malware system in the victim network. Based on the observed patterns, we advocate organizations to train users about cybersecurity best practices, introduce immutable operating systems with limited functionalities, and enforce multi-user authentications. Moreover, we advocate practitioners to leverage the automated mining capability of ChronoCTI and design countermeasures against the recurring attack patterns.



## **23. Safety and Performance, Why Not Both? Bi-Objective Optimized Model Compression against Heterogeneous Attacks Toward AI Software Deployment**

cs.AI

Accepted by IEEE Transactions on Software Engineering (TSE).  Camera-ready Version. arXiv admin note: substantial text overlap with  arXiv:2208.05969

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00996v1) [paper-pdf](http://arxiv.org/pdf/2401.00996v1)

**Authors**: Jie Zhu, Leye Wang, Xiao Han, Anmin Liu, Tao Xie

**Abstract**: The size of deep learning models in artificial intelligence (AI) software is increasing rapidly, hindering the large-scale deployment on resource-restricted devices (e.g., smartphones). To mitigate this issue, AI software compression plays a crucial role, which aims to compress model size while keeping high performance. However, the intrinsic defects in a big model may be inherited by the compressed one. Such defects may be easily leveraged by adversaries, since a compressed model is usually deployed in a large number of devices without adequate protection. In this article, we aim to address the safe model compression problem from the perspective of safety-performance co-optimization. Specifically, inspired by the test-driven development (TDD) paradigm in software engineering, we propose a test-driven sparse training framework called SafeCompress. By simulating the attack mechanism as safety testing, SafeCompress can automatically compress a big model to a small one following the dynamic sparse training paradigm. Then, considering two kinds of representative and heterogeneous attack mechanisms, i.e., black-box membership inference attack and white-box membership inference attack, we develop two concrete instances called BMIA-SafeCompress and WMIA-SafeCompress. Further, we implement another instance called MMIA-SafeCompress by extending SafeCompress to defend against the occasion when adversaries conduct black-box and white-box membership inference attacks simultaneously. We conduct extensive experiments on five datasets for both computer vision and natural language processing tasks. The results show the effectiveness and generalizability of our framework. We also discuss how to adapt SafeCompress to other attacks besides membership inference attack, demonstrating the flexibility of SafeCompress.



## **24. Detection and Defense Against Prominent Attacks on Preconditioned LLM-Integrated Virtual Assistants**

cs.CR

Accepted to be published in the Proceedings of the 10th IEEE CSDE  2023, the Asia-Pacific Conference on Computer Science and Data Engineering  2023

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00994v1) [paper-pdf](http://arxiv.org/pdf/2401.00994v1)

**Authors**: Chun Fai Chan, Daniel Wankit Yip, Aysan Esmradi

**Abstract**: The emergence of LLM (Large Language Model) integrated virtual assistants has brought about a rapid transformation in communication dynamics. During virtual assistant development, some developers prefer to leverage the system message, also known as an initial prompt or custom prompt, for preconditioning purposes. However, it is important to recognize that an excessive reliance on this functionality raises the risk of manipulation by malicious actors who can exploit it with carefully crafted prompts. Such malicious manipulation poses a significant threat, potentially compromising the accuracy and reliability of the virtual assistant's responses. Consequently, safeguarding the virtual assistants with detection and defense mechanisms becomes of paramount importance to ensure their safety and integrity. In this study, we explored three detection and defense mechanisms aimed at countering attacks that target the system message. These mechanisms include inserting a reference key, utilizing an LLM evaluator, and implementing a Self-Reminder. To showcase the efficacy of these mechanisms, they were tested against prominent attack techniques. Our findings demonstrate that the investigated mechanisms are capable of accurately identifying and counteracting the attacks. The effectiveness of these mechanisms underscores their potential in safeguarding the integrity and reliability of virtual assistants, reinforcing the importance of their implementation in real-world scenarios. By prioritizing the security of virtual assistants, organizations can maintain user trust, preserve the integrity of the application, and uphold the high standards expected in this era of transformative technologies.



## **25. A Novel Evaluation Framework for Assessing Resilience Against Prompt Injection Attacks in Large Language Models**

cs.CR

Accepted to be published in the Proceedings of The 10th IEEE CSDE  2023, the Asia-Pacific Conference on Computer Science and Data Engineering  2023

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00991v1) [paper-pdf](http://arxiv.org/pdf/2401.00991v1)

**Authors**: Daniel Wankit Yip, Aysan Esmradi, Chun Fai Chan

**Abstract**: Prompt injection attacks exploit vulnerabilities in large language models (LLMs) to manipulate the model into unintended actions or generate malicious content. As LLM integrated applications gain wider adoption, they face growing susceptibility to such attacks. This study introduces a novel evaluation framework for quantifying the resilience of applications. The framework incorporates innovative techniques designed to ensure representativeness, interpretability, and robustness. To ensure the representativeness of simulated attacks on the application, a meticulous selection process was employed, resulting in 115 carefully chosen attacks based on coverage and relevance. For enhanced interpretability, a second LLM was utilized to evaluate the responses generated from these simulated attacks. Unlike conventional malicious content classifiers that provide only a confidence score, the LLM-based evaluation produces a score accompanied by an explanation, thereby enhancing interpretability. Subsequently, a resilience score is computed by assigning higher weights to attacks with greater impact, thus providing a robust measurement of the application resilience. To assess the framework's efficacy, it was applied on two LLMs, namely Llama2 and ChatGLM. Results revealed that Llama2, the newer model exhibited higher resilience compared to ChatGLM. This finding substantiates the effectiveness of the framework, aligning with the prevailing notion that newer models tend to possess greater resilience. Moreover, the framework exhibited exceptional versatility, requiring only minimal adjustments to accommodate emerging attack techniques and classifications, thereby establishing itself as an effective and practical solution. Overall, the framework offers valuable insights that empower organizations to make well-informed decisions to fortify their applications against potential threats from prompt injection.



## **26. Opening A Pandora's Box: Things You Should Know in the Era of Custom GPTs**

cs.CR

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2401.00905v1) [paper-pdf](http://arxiv.org/pdf/2401.00905v1)

**Authors**: Guanhong Tao, Siyuan Cheng, Zhuo Zhang, Junmin Zhu, Guangyu Shen, Xiangyu Zhang

**Abstract**: The emergence of large language models (LLMs) has significantly accelerated the development of a wide range of applications across various fields. There is a growing trend in the construction of specialized platforms based on LLMs, such as the newly introduced custom GPTs by OpenAI. While custom GPTs provide various functionalities like web browsing and code execution, they also introduce significant security threats. In this paper, we conduct a comprehensive analysis of the security and privacy issues arising from the custom GPT platform. Our systematic examination categorizes potential attack scenarios into three threat models based on the role of the malicious actor, and identifies critical data exchange channels in custom GPTs. Utilizing the STRIDE threat modeling framework, we identify 26 potential attack vectors, with 19 being partially or fully validated in real-world settings. Our findings emphasize the urgent need for robust security and privacy measures in the custom GPT ecosystem, especially in light of the forthcoming launch of the official GPT store by OpenAI.



## **27. Identifying and Mitigating the Security Risks of Generative AI**

cs.AI

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2308.14840v4) [paper-pdf](http://arxiv.org/pdf/2308.14840v4)

**Authors**: Clark Barrett, Brad Boyd, Elie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang

**Abstract**: Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.



## **28. Task Contamination: Language Models May Not Be Few-Shot Anymore**

cs.CL

Accepted by AAAI 2024

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.16337v1) [paper-pdf](http://arxiv.org/pdf/2312.16337v1)

**Authors**: Changmao Li, Jeffrey Flanigan

**Abstract**: Large language models (LLMs) offer impressive performance in various zero-shot and few-shot tasks. However, their success in zero-shot and few-shot settings may be affected by task contamination, a potential limitation that has not been thoroughly examined. This paper investigates how zero-shot and few-shot performance of LLMs has changed chronologically over time. Utilizing GPT-3 series models and several other recent open-sourced LLMs, and controlling for dataset difficulty, we find that on datasets released before the LLM training data creation date, LLMs perform surprisingly better than on datasets released after. This strongly indicates that, for many LLMs, there exists task contamination on zero-shot and few-shot evaluation for datasets released prior to the LLMs' training data creation date. Additionally, we utilize training data inspection, task example extraction, and a membership inference attack, which reveal further evidence of task contamination. Importantly, we find that for classification tasks with no possibility of task contamination, LLMs rarely demonstrate statistically significant improvements over simple majority baselines, in both zero and few-shot settings.



## **29. Vulnerability of Machine Learning Approaches Applied in IoT-based Smart Grid: A Review**

cs.CR

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2308.15736v3) [paper-pdf](http://arxiv.org/pdf/2308.15736v3)

**Authors**: Zhenyong Zhang, Mengxiang Liu, Mingyang Sun, Ruilong Deng, Peng Cheng, Dusit Niyato, Mo-Yuen Chow, Jiming Chen

**Abstract**: Machine learning (ML) sees an increasing prevalence of being used in the internet-of-things (IoT)-based smart grid. However, the trustworthiness of ML is a severe issue that must be addressed to accommodate the trend of ML-based smart grid applications (MLsgAPPs). The adversarial distortion injected into the power signal will greatly affect the system's normal control and operation. Therefore, it is imperative to conduct vulnerability assessment for MLsgAPPs applied in the context of safety-critical power systems. In this paper, we provide a comprehensive review of the recent progress in designing attack and defense methods for MLsgAPPs. Unlike the traditional survey about ML security, this is the first review work about the security of MLsgAPPs that focuses on the characteristics of power systems. We first highlight the specifics for constructing the adversarial attacks on MLsgAPPs. Then, the vulnerability of MLsgAPP is analyzed from both the aspects of the power system and ML model. Afterward, a comprehensive survey is conducted to review and compare existing studies about the adversarial attacks on MLsgAPPs in scenarios of generation, transmission, distribution, and consumption, and the countermeasures are reviewed according to the attacks that they defend against. Finally, the future research directions are discussed on the attacker's and defender's side, respectively. We also analyze the potential vulnerability of large language model-based (e.g., ChatGPT) power system applications. Overall, we encourage more researchers to contribute to investigating the adversarial issues of MLsgAPPs.



## **30. From Shortcuts to Triggers: Backdoor Defense with Denoised PoE**

cs.CL

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2305.14910v2) [paper-pdf](http://arxiv.org/pdf/2305.14910v2)

**Authors**: Qin Liu, Fei Wang, Chaowei Xiao, Muhao Chen

**Abstract**: Language models are often at risk of diverse backdoor attacks, especially data poisoning. Thus, it is important to investigate defense solutions for addressing them. Existing backdoor defense methods mainly focus on backdoor attacks with explicit triggers, leaving a universal defense against various backdoor attacks with diverse triggers largely unexplored. In this paper, we propose an end-to-end ensemble-based backdoor defense framework, DPoE (Denoised Product-of-Experts), which is inspired by the shortcut nature of backdoor attacks, to defend various backdoor attacks. DPoE consists of two models: a shallow model that captures the backdoor shortcuts and a main model that is prevented from learning the backdoor shortcuts. To address the label flip caused by backdoor attackers, DPoE incorporates a denoising design. Experiments on SST-2 dataset show that DPoE significantly improves the defense performance against various types of backdoor triggers including word-level, sentence-level, and syntactic triggers. Furthermore, DPoE is also effective under a more challenging but practical setting that mixes multiple types of trigger.



## **31. A Mutation-Based Method for Multi-Modal Jailbreaking Attack Detection**

cs.CR

12 pages, 8 figures

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.10766v2) [paper-pdf](http://arxiv.org/pdf/2312.10766v2)

**Authors**: Xiaoyu Zhang, Cen Zhang, Tianlin Li, Yihao Huang, Xiaojun Jia, Xiaofei Xie, Yang Liu, Chao Shen

**Abstract**: Large Language Models and Multi-Modal LLMs have become pervasive, and so does the importance of their security; yet, modern LLMs are known to be vulnerable to jailbreaking attacks. These attacks can allow malicious users to exploit the models, making the case for effective jailbreak detection mechanisms an essential aspect of maintaining the integrity and trustworthiness of LLM-based applications. However, existing detection works on jailbreak attacks have limitations. Existing post-query-based strategies require target domain knowledge, and pre-query-based methods mainly focus on text-level attacks and fail to meet the increasingly complex multi-modal security requirements placed upon contemporary LLMs. This gap underscores the need for a more comprehensive approach to safeguarding these influential systems.   In this work, we propose JailGuard, the first mutation-based jailbreaking detection framework which supports both image and text modalities. Our key observation is that attack queries inherently possess less robustness compared to benign queries. Specifically, to confuse the model, attack queries are usually crafted with well-designed templates or complicate perturbations, leading to a fact that a slight disturbance in input may result in a drastic change in the response. This lack of robustness can be utilized in attack detection. Based on this intuition, we designed and implemented a detection framework comprising 19 different mutators and a divergence-based detection formula. To fully understand the effectiveness of our framework, we built the first multi-modal LLM jailbreaking attack dataset, which has 304 items of data, covering ten types of known jailbreaking attacks on image and text modalities. The evaluation suggests that JailGuard achieves the best detection accuracy of 89.38%/85.42% on image and text inputs, outperforming state-of-the-art defense methods by 15.28%.



## **32. A Survey on Large Language Models for Software Engineering**

cs.SE

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.15223v1) [paper-pdf](http://arxiv.org/pdf/2312.15223v1)

**Authors**: Quanjun Zhang, Chunrong Fang, Yang Xie, Yaxin Zhang, Yun Yang, Weisong Sun, Shengcheng Yu, Zhenyu Chen

**Abstract**: Software Engineering (SE) is the systematic design, development, and maintenance of software applications, underpinning the digital infrastructure of our modern mainworld. Very recently, the SE community has seen a rapidly increasing number of techniques employing Large Language Models (LLMs) to automate a broad range of SE tasks. Nevertheless, existing information of the applications, effects, and possible limitations of LLMs within SE is still not well-studied.   In this paper, we provide a systematic survey to summarize the current state-of-the-art research in the LLM-based SE community. We summarize 30 representative LLMs of Source Code across three model architectures, 15 pre-training objectives across four categories, and 16 downstream tasks across five categories. We then present a detailed summarization of the recent SE studies for which LLMs are commonly utilized, including 155 studies for 43 specific code-related tasks across four crucial phases within the SE workflow. Besides, we summarize existing attempts to empirically evaluate LLMs in SE, such as benchmarks, empirical studies, and exploration of SE education. We also discuss several critical aspects of optimization and applications of LLMs in SE, such as security attacks, model tuning, and model compression. Finally, we highlight several challenges and potential opportunities on applying LLMs for future SE studies, such as exploring domain LLMs and constructing clean evaluation datasets. Overall, our work can help researchers gain a comprehensive understanding about the achievements of the existing LLM-based SE studies and promote the practical application of these techniques. Our artifacts are publicly available and will continuously updated at the living repository: \url{https://github.com/iSEngLab/AwesomeLLM4SE}.



## **33. Spear Phishing With Large Language Models**

cs.CY

16 pages, 10 figures

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2305.06972v3) [paper-pdf](http://arxiv.org/pdf/2305.06972v3)

**Authors**: Julian Hazell

**Abstract**: Recent progress in artificial intelligence (AI), particularly in the domain of large language models (LLMs), has resulted in powerful and versatile dual-use systems. This intelligence can be put towards a wide variety of beneficial tasks, yet it can also be used to cause harm. This study explores one such harm by examining how LLMs can be used for spear phishing, a form of cybercrime that involves manipulating targets into divulging sensitive information. I first explore LLMs' ability to assist with the reconnaissance and message generation stages of a spear phishing attack, where I find that LLMs are capable of assisting with the email generation phase of a spear phishing attack. To explore how LLMs could potentially be harnessed to scale spear phishing campaigns, I then create unique spear phishing messages for over 600 British Members of Parliament using OpenAI's GPT-3.5 and GPT-4 models. My findings provide some evidence that these messages are not only realistic but also cost-effective, with each email costing only a fraction of a cent to generate. Next, I demonstrate how basic prompt engineering can circumvent safeguards installed in LLMs, highlighting the need for further research into robust interventions that can help prevent models from being misused. To further address these evolving risks, I explore two potential solutions: structured access schemes, such as application programming interfaces, and LLM-based defensive systems.



## **34. MetaAID 2.5: A Secure Framework for Developing Metaverse Applications via Large Language Models**

cs.CR

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.14480v1) [paper-pdf](http://arxiv.org/pdf/2312.14480v1)

**Authors**: Hongyin Zhu

**Abstract**: Large language models (LLMs) are increasingly being used in Metaverse environments to generate dynamic and realistic content and to control the behavior of non-player characters (NPCs). However, the cybersecurity concerns associated with LLMs have become increasingly prominent. Previous research has primarily focused on patching system vulnerabilities to enhance cybersecurity, but these approaches are not well-suited to the Metaverse, where the virtual space is more complex, LLMs are vulnerable, and ethical user interaction is critical. Moreover, the scope of cybersecurity in the Metaverse is expected to expand significantly. This paper proposes a method for enhancing cybersecurity through the simulation of user interaction with LLMs. Our goal is to educate users and strengthen their defense capabilities through exposure to a comprehensive simulation system. This system includes extensive Metaverse cybersecurity Q&A and attack simulation scenarios. By engaging with these, users will improve their ability to recognize and withstand risks. Additionally, to address the ethical implications of user input, we propose using LLMs as evaluators to assess user content across five dimensions. We further adapt the models through vocabulary expansion training to better understand personalized inputs and emoticons. We conduct experiments on multiple LLMs and find that our approach is effective.



## **35. HW-V2W-Map: Hardware Vulnerability to Weakness Mapping Framework for Root Cause Analysis with GPT-assisted Mitigation Suggestion**

cs.CR

22 pages, 10 pages appendix, 10 figures, Submitted to ACM TODAES

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13530v1) [paper-pdf](http://arxiv.org/pdf/2312.13530v1)

**Authors**: Yu-Zheng Lin, Muntasir Mamun, Muhtasim Alam Chowdhury, Shuyu Cai, Mingyu Zhu, Banafsheh Saber Latibari, Kevin Immanuel Gubbi, Najmeh Nazari Bavarsad, Arjun Caputo, Avesta Sasan, Houman Homayoun, Setareh Rafatirad, Pratik Satam, Soheil Salehi

**Abstract**: The escalating complexity of modern computing frameworks has resulted in a surge in the cybersecurity vulnerabilities reported to the National Vulnerability Database (NVD) by practitioners. Despite the fact that the stature of NVD is one of the most significant databases for the latest insights into vulnerabilities, extracting meaningful trends from such a large amount of unstructured data is still challenging without the application of suitable technological methodologies. Previous efforts have mostly concentrated on software vulnerabilities; however, a holistic strategy incorporates approaches for mitigating vulnerabilities, score prediction, and a knowledge-generating system that may extract relevant insights from the Common Weakness Enumeration (CWE) and Common Vulnerability Exchange (CVE) databases is notably absent. As the number of hardware attacks on Internet of Things (IoT) devices continues to rapidly increase, we present the Hardware Vulnerability to Weakness Mapping (HW-V2W-Map) Framework, which is a Machine Learning (ML) framework focusing on hardware vulnerabilities and IoT security. The architecture that we have proposed incorporates an Ontology-driven Storytelling framework, which automates the process of updating the ontology in order to recognize patterns and evolution of vulnerabilities over time and provides approaches for mitigating the vulnerabilities. The repercussions of vulnerabilities can be mitigated as a result of this, and conversely, future exposures can be predicted and prevented. Furthermore, our proposed framework utilized Generative Pre-trained Transformer (GPT) Large Language Models (LLMs) to provide mitigation suggestions.



## **36. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

cs.CL

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14197v1) [paper-pdf](http://arxiv.org/pdf/2312.14197v1)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Keegan Hines, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: Recent remarkable advancements in large language models (LLMs) have led to their widespread adoption in various applications. A key feature of these applications is the combination of LLMs with external content, where user instructions and third-party content are combined to create prompts for LLM processing. These applications, however, are vulnerable to indirect prompt injection attacks, where malicious instructions embedded within external content compromise LLM's output, causing their responses to deviate from user expectations. Despite the discovery of this security issue, no comprehensive analysis of indirect prompt injection attacks on different LLMs is available due to the lack of a benchmark. Furthermore, no effective defense has been proposed.   In this work, we introduce the first benchmark, BIPIA, to measure the robustness of various LLMs and defenses against indirect prompt injection attacks. Our experiments reveal that LLMs with greater capabilities exhibit more vulnerable to indirect prompt injection attacks for text tasks, resulting in a higher ASR. We hypothesize that indirect prompt injection attacks are mainly due to the LLMs' inability to distinguish between instructions and external content. Based on this conjecture, we propose four black-box methods based on prompt learning and a white-box defense methods based on fine-tuning with adversarial training to enable LLMs to distinguish between instructions and external content and ignore instructions in the external content. Our experimental results show that our black-box defense methods can effectively reduce ASR but cannot completely thwart indirect prompt injection attacks, while our white-box defense method can reduce ASR to nearly zero with little adverse impact on the LLM's performance on general tasks. We hope that our benchmark and defenses can inspire future work in this important area.



## **37. Universal and Transferable Adversarial Attacks on Aligned Language Models**

cs.CL

Website: http://llm-attacks.org/

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2307.15043v2) [paper-pdf](http://arxiv.org/pdf/2307.15043v2)

**Authors**: Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson

**Abstract**: Because "out-of-the-box" large language models are capable of generating a great deal of objectionable content, recent work has focused on aligning these models in an attempt to prevent undesirable generation. While there has been some success at circumventing these measures -- so-called "jailbreaks" against LLMs -- these attacks have required significant human ingenuity and are brittle in practice. In this paper, we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries for an LLM to produce objectionable content, aims to maximize the probability that the model produces an affirmative response (rather than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques, and also improves over past automatic prompt generation methods.   Surprisingly, we find that the adversarial prompts generated by our approach are quite transferable, including to black-box, publicly released LLMs. Specifically, we train an adversarial attack suffix on multiple prompts (i.e., queries asking for many different types of objectionable content), as well as multiple models (in our case, Vicuna-7B and 13B). When doing so, the resulting attack suffix is able to induce objectionable content in the public interfaces to ChatGPT, Bard, and Claude, as well as open source LLMs such as LLaMA-2-Chat, Pythia, Falcon, and others. In total, this work significantly advances the state-of-the-art in adversarial attacks against aligned language models, raising important questions about how such systems can be prevented from producing objectionable information. Code is available at github.com/llm-attacks/llm-attacks.



## **38. Robust Contrastive Language-Image Pre-training against Data Poisoning and Backdoor Attacks**

cs.CV

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2303.06854v2) [paper-pdf](http://arxiv.org/pdf/2303.06854v2)

**Authors**: Wenhan Yang, Jingdong Gao, Baharan Mirzasoleiman

**Abstract**: Contrastive vision-language representation learning has achieved state-of-the-art performance for zero-shot classification, by learning from millions of image-caption pairs crawled from the internet. However, the massive data that powers large multimodal models such as CLIP, makes them extremely vulnerable to various types of targeted data poisoning and backdoor attacks. Despite this vulnerability, robust contrastive vision-language pre-training against such attacks has remained unaddressed. In this work, we propose ROCLIP, the first effective method for robust pre-training multimodal vision-language models against targeted data poisoning and backdoor attacks. ROCLIP effectively breaks the association between poisoned image-caption pairs by considering a relatively large and varying pool of random captions, and matching every image with the text that is most similar to it in the pool instead of its own caption, every few epochs.It also leverages image and text augmentations to further strengthen the defense and improve the performance of the model. Our extensive experiments show that ROCLIP renders state-of-the-art targeted data poisoning and backdoor attacks ineffective during pre-training CLIP models. In particular, ROCLIP decreases the success rate for targeted data poisoning attacks from 93.75% to 12.5% and that of backdoor attacks down to 0%, while improving the model's linear probe performance by 10% and maintains a similar zero shot performance compared to CLIP. By increasing the frequency of matching, ROCLIP is able to defend strong attacks, which add up to 1% poisoned examples to the data, and successfully maintain a low attack success rate of 12.5%, while trading off the performance on some tasks.



## **39. PoisonPrompt: Backdoor Attack on Prompt-based Large Language Models**

cs.CL

To Appear in IEEE ICASSP 2024, code is available at:  https://github.com/grasses/PoisonPrompt

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2310.12439v2) [paper-pdf](http://arxiv.org/pdf/2310.12439v2)

**Authors**: Hongwei Yao, Jian Lou, Zhan Qin

**Abstract**: Prompts have significantly improved the performance of pretrained Large Language Models (LLMs) on various downstream tasks recently, making them increasingly indispensable for a diverse range of LLM application scenarios. However, the backdoor vulnerability, a serious security threat that can maliciously alter the victim model's normal predictions, has not been sufficiently explored for prompt-based LLMs. In this paper, we present POISONPROMPT, a novel backdoor attack capable of successfully compromising both hard and soft prompt-based LLMs. We evaluate the effectiveness, fidelity, and robustness of POISONPROMPT through extensive experiments on three popular prompt methods, using six datasets and three widely used LLMs. Our findings highlight the potential security threats posed by backdoor attacks on prompt-based LLMs and emphasize the need for further research in this area.



## **40. A Comprehensive Survey of Attack Techniques, Implementation, and Mitigation Strategies in Large Language Models**

cs.CR

Accepted to be published in the Proceedings of the 3rd International  Conference on Ubiquitous Security 2023 (UbiSec-2023)

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.10982v1) [paper-pdf](http://arxiv.org/pdf/2312.10982v1)

**Authors**: Aysan Esmradi, Daniel Wankit Yip, Chun Fai Chan

**Abstract**: Ensuring the security of large language models (LLMs) is an ongoing challenge despite their widespread popularity. Developers work to enhance LLMs security, but vulnerabilities persist, even in advanced versions like GPT-4. Attackers exploit these weaknesses, highlighting the need for proactive cybersecurity measures in AI model development. This article explores two attack categories: attacks on models themselves and attacks on model applications. The former requires expertise, access to model data, and significant implementation time, while the latter is more accessible to attackers and has seen increased attention. Our study reviews over 100 recent research works, providing an in-depth analysis of each attack type. We identify the latest attack methods and explore various approaches to carry them out. We thoroughly investigate mitigation techniques, assessing their effectiveness and limitations. Furthermore, we summarize future defenses against these attacks. We also examine real-world techniques, including reported and our implemented attacks on LLMs, to consolidate our findings. Our research highlights the urgency of addressing security concerns and aims to enhance the understanding of LLM attacks, contributing to robust defense development in this evolving domain.



## **41. No-Skim: Towards Efficiency Robustness Evaluation on Skimming-based Language Models**

cs.CR

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.09494v2) [paper-pdf](http://arxiv.org/pdf/2312.09494v2)

**Authors**: Shengyao Zhang, Mi Zhang, Xudong Pan, Min Yang

**Abstract**: To reduce the computation cost and the energy consumption in large language models (LLM), skimming-based acceleration dynamically drops unimportant tokens of the input sequence progressively along layers of the LLM while preserving the tokens of semantic importance. However, our work for the first time reveals the acceleration may be vulnerable to Denial-of-Service (DoS) attacks. In this paper, we propose No-Skim, a general framework to help the owners of skimming-based LLM to understand and measure the robustness of their acceleration scheme. Specifically, our framework searches minimal and unnoticeable perturbations at character-level and token-level to generate adversarial inputs that sufficiently increase the remaining token ratio, thus increasing the computation cost and energy consumption. We systematically evaluate the vulnerability of the skimming acceleration in various LLM architectures including BERT and RoBERTa on the GLUE benchmark. In the worst case, the perturbation found by No-Skim substantially increases the running cost of LLM by over 145% on average. Moreover, No-Skim extends the evaluation framework to various scenarios, making the evaluation conductible with different level of knowledge.



## **42. Privacy-Aware Document Visual Question Answering**

cs.CV

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.10108v1) [paper-pdf](http://arxiv.org/pdf/2312.10108v1)

**Authors**: Rubèn Tito, Khanh Nguyen, Marlon Tobaben, Raouf Kerkouche, Mohamed Ali Souibgui, Kangsoo Jung, Lei Kang, Ernest Valveny, Antti Honkela, Mario Fritz, Dimosthenis Karatzas

**Abstract**: Document Visual Question Answering (DocVQA) is a fast growing branch of document understanding. Despite the fact that documents contain sensitive or copyrighted information, none of the current DocVQA methods offers strong privacy guarantees.   In this work, we explore privacy in the domain of DocVQA for the first time. We highlight privacy issues in state of the art multi-modal LLM models used for DocVQA, and explore possible solutions.   Specifically, we focus on the invoice processing use case as a realistic, widely used scenario for document understanding, and propose a large scale DocVQA dataset comprising invoice documents and associated questions and answers. We employ a federated learning scheme, that reflects the real-life distribution of documents in different businesses, and we explore the use case where the ID of the invoice issuer is the sensitive information to be protected.   We demonstrate that non-private models tend to memorise, behaviour that can lead to exposing private information. We then evaluate baseline training schemes employing federated learning and differential privacy in this multi-modal scenario, where the sensitive information might be exposed through any of the two input modalities: vision (document image) or language (OCR tokens).   Finally, we design an attack exploiting the memorisation effect of the model, and demonstrate its effectiveness in probing different DocVQA models.



## **43. AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models**

cs.CR

Version 2 updates: Added comparison of three more evaluation methods  and their reliability check using human labeling. Added results for  jailbreaking Llama2 (individual behavior) and included complexity and  hyperparameter analysis. Revised objectives for prompt leaking. Other minor  changes made

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2310.15140v2) [paper-pdf](http://arxiv.org/pdf/2310.15140v2)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent studies suggest that defending against these attacks is possible: adversarial attacks generate unlimited but unreadable gibberish prompts, detectable by perplexity-based filters; manual jailbreak attacks craft readable prompts, but their limited number due to the necessity of human creativity allows for easy blocking. In this paper, we show that these solutions may be too optimistic. We introduce AutoDAN, an interpretable, gradient-based adversarial attack that merges the strengths of both attack types. Guided by the dual goals of jailbreak and readability, AutoDAN optimizes and generates tokens one by one from left to right, resulting in readable prompts that bypass perplexity filters while maintaining high attack success rates. Notably, these prompts, generated from scratch using gradients, are interpretable and diverse, with emerging strategies commonly seen in manual jailbreak attacks. They also generalize to unforeseen harmful behaviors and transfer to black-box LLMs better than their unreadable counterparts when using limited training data or a single proxy model. Furthermore, we show the versatility of AutoDAN by automatically leaking system prompts using a customized objective. Our work offers a new way to red-team LLMs and understand jailbreak mechanisms via interpretability.



## **44. FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts**

cs.CR

Technical Report

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2311.05608v2) [paper-pdf](http://arxiv.org/pdf/2311.05608v2)

**Authors**: Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, Xiaoyun Wang

**Abstract**: Ensuring the safety of artificial intelligence-generated content (AIGC) is a longstanding topic in the artificial intelligence (AI) community, and the safety concerns associated with Large Language Models (LLMs) have been widely investigated. Recently, large vision-language models (VLMs) represent an unprecedented revolution, as they are built upon LLMs but can incorporate additional modalities (e.g., images). However, the safety of VLMs lacks systematic evaluation, and there may be an overconfidence in the safety guarantees provided by their underlying LLMs. In this paper, to demonstrate that introducing additional modality modules leads to unforeseen AI safety issues, we propose FigStep, a straightforward yet effective jailbreaking algorithm against VLMs. Instead of feeding textual harmful instructions directly, FigStep converts the harmful content into images through typography to bypass the safety alignment within the textual module of the VLMs, inducing VLMs to output unsafe responses that violate common AI safety policies. In our evaluation, we manually review 46,500 model responses generated by 3 families of the promising open-source VLMs, i.e., LLaVA, MiniGPT4, and CogVLM (a total of 6 VLMs). The experimental results show that FigStep can achieve an average attack success rate of 82.50% on 500 harmful queries in 10 topics. Moreover, we demonstrate that the methodology of FigStep can even jailbreak GPT-4V, which already leverages an OCR detector to filter harmful queries. Above all, our work reveals that VLMs are vulnerable to jailbreaking attacks, which highlights the necessity of novel safety alignments between visual and textual modalities.



## **45. Efficient Representation of the Activation Space in Deep Neural Networks**

cs.LG

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08143v1) [paper-pdf](http://arxiv.org/pdf/2312.08143v1)

**Authors**: Tanya Akumu, Celia Cintas, Girmaw Abebe Tadesse, Adebayo Oshingbesan, Skyler Speakman, Edward McFowland III

**Abstract**: The representations of the activation space of deep neural networks (DNNs) are widely utilized for tasks like natural language processing, anomaly detection and speech recognition. Due to the diverse nature of these tasks and the large size of DNNs, an efficient and task-independent representation of activations becomes crucial. Empirical p-values have been used to quantify the relative strength of an observed node activation compared to activations created by already-known inputs. Nonetheless, keeping raw data for these calculations increases memory resource consumption and raises privacy concerns. To this end, we propose a model-agnostic framework for creating representations of activations in DNNs using node-specific histograms to compute p-values of observed activations without retaining already-known inputs. Our proposed approach demonstrates promising potential when validated with multiple network architectures across various downstream tasks and compared with the kernel density estimates and brute-force empirical baselines. In addition, the framework reduces memory usage by 30% with up to 4 times faster p-value computing time while maintaining state of-the-art detection power in downstream tasks such as the detection of adversarial attacks and synthesized content. Moreover, as we do not persist raw data at inference time, we could potentially reduce susceptibility to attacks and privacy issues.



## **46. Causality Analysis for Evaluating the Security of Large Language Models**

cs.AI

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07876v1) [paper-pdf](http://arxiv.org/pdf/2312.07876v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) such as GPT and Llama2 are increasingly adopted in many safety-critical applications. Their security is thus essential. Even with considerable efforts spent on reinforcement learning from human feedback (RLHF), recent studies have shown that LLMs are still subject to attacks such as adversarial perturbation and Trojan attacks. Further research is thus needed to evaluate their security and/or understand the lack of it. In this work, we propose a framework for conducting light-weight causality-analysis of LLMs at the token, layer, and neuron level. We applied our framework to open-source LLMs such as Llama2 and Vicuna and had multiple interesting discoveries. Based on a layer-level causality analysis, we show that RLHF has the effect of overfitting a model to harmful prompts. It implies that such security can be easily overcome by `unusual' harmful prompts. As evidence, we propose an adversarial perturbation method that achieves 100\% attack success rate on the red-teaming tasks of the Trojan Detection Competition 2023. Furthermore, we show the existence of one mysterious neuron in both Llama2 and Vicuna that has an unreasonably high causal effect on the output. While we are uncertain on why such a neuron exists, we show that it is possible to conduct a ``Trojan'' attack targeting that particular neuron to completely cripple the LLM, i.e., we can generate transferable suffixes to prompts that frequently make the LLM produce meaningless responses.



## **47. DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions**

cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.04730v2) [paper-pdf](http://arxiv.org/pdf/2312.04730v2)

**Authors**: Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao

**Abstract**: With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.



## **48. Maatphor: Automated Variant Analysis for Prompt Injection Attacks**

cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.11513v1) [paper-pdf](http://arxiv.org/pdf/2312.11513v1)

**Authors**: Ahmed Salem, Andrew Paverd, Boris Köpf

**Abstract**: Prompt injection has emerged as a serious security threat to large language models (LLMs). At present, the current best-practice for defending against newly-discovered prompt injection techniques is to add additional guardrails to the system (e.g., by updating the system prompt or using classifiers on the input and/or output of the model.) However, in the same way that variants of a piece of malware are created to evade anti-virus software, variants of a prompt injection can be created to evade the LLM's guardrails. Ideally, when a new prompt injection technique is discovered, candidate defenses should be tested not only against the successful prompt injection, but also against possible variants.   In this work, we present, a tool to assist defenders in performing automated variant analysis of known prompt injection attacks. This involves solving two main challenges: (1) automatically generating variants of a given prompt according, and (2) automatically determining whether a variant was effective based only on the output of the model. This tool can also assist in generating datasets for jailbreak and prompt injection attacks, thus overcoming the scarcity of data in this domain.   We evaluate Maatphor on three different types of prompt injection tasks. Starting from an ineffective (0%) seed prompt, Maatphor consistently generates variants that are at least 60% effective within the first 40 iterations.



## **49. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

cs.CL

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2311.06062v2) [paper-pdf](http://arxiv.org/pdf/2311.06062v2)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.



## **50. Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack**

cs.CL

17 pages,10 figures

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06924v1) [paper-pdf](http://arxiv.org/pdf/2312.06924v1)

**Authors**: Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong

**Abstract**: Recent developments in balancing the usefulness and safety of Large Language Models (LLMs) have raised a critical question: Are mainstream NLP tasks adequately aligned with safety consideration? Our study, focusing on safety-sensitive documents obtained through adversarial attacks, reveals significant disparities in the safety alignment of various NLP tasks. For instance, LLMs can effectively summarize malicious long documents but often refuse to translate them. This discrepancy highlights a previously unidentified vulnerability: attacks exploiting tasks with weaker safety alignment, like summarization, can potentially compromise the integraty of tasks traditionally deemed more robust, such as translation and question-answering (QA). Moreover, the concurrent use of multiple NLP tasks with lesser safety alignment increases the risk of LLMs inadvertently processing harmful content. We demonstrate these vulnerabilities in various safety-aligned LLMs, particularly Llama2 models and GPT-4, indicating an urgent need for strengthening safety alignments across a broad spectrum of NLP tasks.



