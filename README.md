# Latest Adversarial Attack Papers
**update at 2023-12-02 11:30:07**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Adversarial Attacks and Defenses for Wireless Signal Classifiers using CDI-aware GANs**

cs.IT

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.18820v1) [paper-pdf](http://arxiv.org/pdf/2311.18820v1)

**Authors**: Sujata Sinha, Alkan Soysal

**Abstract**: We introduce a Channel Distribution Information (CDI)-aware Generative Adversarial Network (GAN), designed to address the unique challenges of adversarial attacks in wireless communication systems. The generator in this CDI-aware GAN maps random input noise to the feature space, generating perturbations intended to deceive a target modulation classifier. Its discriminators play a dual role: one enforces that the perturbations follow a Gaussian distribution, making them indistinguishable from Gaussian noise, while the other ensures these perturbations account for realistic channel effects and resemble no-channel perturbations.   Our proposed CDI-aware GAN can be used as an attacker and a defender. In attack scenarios, the CDI-aware GAN demonstrates its prowess by generating robust adversarial perturbations that effectively deceive the target classifier, outperforming known methods. Furthermore, CDI-aware GAN as a defender significantly improves the target classifier's resilience against adversarial attacks.



## **2. Differentiable JPEG: The Devil is in the Details**

cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2309.06978v3) [paper-pdf](http://arxiv.org/pdf/2309.06978v3)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.



## **3. Diffusion Models for Imperceptible and Transferable Adversarial Attack**

cs.CV

Code Page: https://github.com/WindVChen/DiffAttack. In Paper Version  v2, we incorporate more discussions and experiments

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2305.08192v2) [paper-pdf](http://arxiv.org/pdf/2305.08192v2)

**Authors**: Jianqi Chen, Hao Chen, Keyan Chen, Yilan Zhang, Zhengxia Zou, Zhenwei Shi

**Abstract**: Many existing adversarial attacks generate $L_p$-norm perturbations on image RGB space. Despite some achievements in transferability and attack success rate, the crafted adversarial examples are easily perceived by human eyes. Towards visual imperceptibility, some recent works explore unrestricted attacks without $L_p$-norm constraints, yet lacking transferability of attacking black-box models. In this work, we propose a novel imperceptible and transferable attack by leveraging both the generative and discriminative power of diffusion models. Specifically, instead of direct manipulation in pixel space, we craft perturbations in the latent space of diffusion models. Combined with well-designed content-preserving structures, we can generate human-insensitive perturbations embedded with semantic clues. For better transferability, we further "deceive" the diffusion model which can be viewed as an implicit recognition surrogate, by distracting its attention away from the target regions. To our knowledge, our proposed method, DiffAttack, is the first that introduces diffusion models into the adversarial attack field. Extensive experiments on various model structures, datasets, and defense methods have demonstrated the superiority of our attack over the existing attack methods.



## **4. Data-Agnostic Model Poisoning against Federated Learning: A Graph Autoencoder Approach**

cs.LG

15 pages, 10 figures, submitted to IEEE Transactions on Information  Forensics and Security (TIFS)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.18498v1) [paper-pdf](http://arxiv.org/pdf/2311.18498v1)

**Authors**: Kai Li, Jingjing Zheng, Xin Yuan, Wei Ni, Ozgur B. Akan, H. Vincent Poor

**Abstract**: This paper proposes a novel, data-agnostic, model poisoning attack on Federated Learning (FL), by designing a new adversarial graph autoencoder (GAE)-based framework. The attack requires no knowledge of FL training data and achieves both effectiveness and undetectability. By listening to the benign local models and the global model, the attacker extracts the graph structural correlations among the benign local models and the training data features substantiating the models. The attacker then adversarially regenerates the graph structural correlations while maximizing the FL training loss, and subsequently generates malicious local models using the adversarial graph structure and the training data features of the benign ones. A new algorithm is designed to iteratively train the malicious local models using GAE and sub-gradient descent. The convergence of FL under attack is rigorously proved, with a considerably large optimality gap. Experiments show that the FL accuracy drops gradually under the proposed attack and existing defense mechanisms fail to detect it. The attack can give rise to an infection across all benign devices, making it a serious threat to FL.



## **5. Towards Safer Generative Language Models: A Survey on Safety Risks, Evaluations, and Improvements**

cs.AI

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2302.09270v3) [paper-pdf](http://arxiv.org/pdf/2302.09270v3)

**Authors**: Jiawen Deng, Jiale Cheng, Hao Sun, Zhexin Zhang, Minlie Huang

**Abstract**: As generative large model capabilities advance, safety concerns become more pronounced in their outputs. To ensure the sustainable growth of the AI ecosystem, it's imperative to undertake a holistic evaluation and refinement of associated safety risks. This survey presents a framework for safety research pertaining to large models, delineating the landscape of safety risks as well as safety evaluation and improvement methods. We begin by introducing safety issues of wide concern, then delve into safety evaluation methods for large models, encompassing preference-based testing, adversarial attack approaches, issues detection, and other advanced evaluation methods. Additionally, we explore the strategies for enhancing large model safety from training to deployment, highlighting cutting-edge safety approaches for each stage in building large models. Finally, we discuss the core challenges in advancing towards more responsible AI, including the interpretability of safety mechanisms, ongoing safety issues, and robustness against malicious attacks. Through this survey, we aim to provide clear technical guidance for safety researchers and encourage further study on the safety of large models.



## **6. On the Robustness of Decision-Focused Learning**

cs.LG

17 pages, 45 figures, submitted to AAAI artificial intelligence for  operations research workshop

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.16487v2) [paper-pdf](http://arxiv.org/pdf/2311.16487v2)

**Authors**: Yehya Farhat

**Abstract**: Decision-Focused Learning (DFL) is an emerging learning paradigm that tackles the task of training a machine learning (ML) model to predict missing parameters of an incomplete optimization problem, where the missing parameters are predicted. DFL trains an ML model in an end-to-end system, by integrating the prediction and optimization tasks, providing better alignment of the training and testing objectives. DFL has shown a lot of promise and holds the capacity to revolutionize decision-making in many real-world applications. However, very little is known about the performance of these models under adversarial attacks. We adopt ten unique DFL methods and benchmark their performance under two distinctly focused attacks adapted towards the Predict-then-Optimize problem setting. Our study proposes the hypothesis that the robustness of a model is highly correlated with its ability to find predictions that lead to optimal decisions without deviating from the ground-truth label. Furthermore, we provide insight into how to target the models that violate this condition and show how these models respond differently depending on the achieved optimality at the end of their training cycles.



## **7. Improving the Robustness of Transformer-based Large Language Models with Dynamic Attention**

cs.CL

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.17400v2) [paper-pdf](http://arxiv.org/pdf/2311.17400v2)

**Authors**: Lujia Shen, Yuwen Pu, Shouling Ji, Changjiang Li, Xuhong Zhang, Chunpeng Ge, Ting Wang

**Abstract**: Transformer-based models, such as BERT and GPT, have been widely adopted in natural language processing (NLP) due to their exceptional performance. However, recent studies show their vulnerability to textual adversarial attacks where the model's output can be misled by intentionally manipulating the text inputs. Despite various methods that have been proposed to enhance the model's robustness and mitigate this vulnerability, many require heavy consumption resources (e.g., adversarial training) or only provide limited protection (e.g., defensive dropout). In this paper, we propose a novel method called dynamic attention, tailored for the transformer architecture, to enhance the inherent robustness of the model itself against various adversarial attacks. Our method requires no downstream task knowledge and does not incur additional costs. The proposed dynamic attention consists of two modules: (I) attention rectification, which masks or weakens the attention value of the chosen tokens, and (ii) dynamic modeling, which dynamically builds the set of candidate tokens. Extensive experiments demonstrate that dynamic attention significantly mitigates the impact of adversarial attacks, improving up to 33\% better performance than previous methods against widely-used adversarial attacks. The model-level design of dynamic attention enables it to be easily combined with other defense methods (e.g., adversarial training) to further enhance the model's robustness. Furthermore, we demonstrate that dynamic attention preserves the state-of-the-art robustness space of the original model compared to other dynamic modeling methods.



## **8. Effective Backdoor Mitigation Depends on the Pre-training Objective**

cs.LG

Accepted for oral presentation at BUGS workshop @ NeurIPS 2023  (https://neurips2023-bugs.github.io/)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.14948v2) [paper-pdf](http://arxiv.org/pdf/2311.14948v2)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for pre-training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in these models such as CleanCLIP which is the current state-of-the-art approach. In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training. We observe that stronger pre-training objectives correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP is ineffective when stronger pre-training objectives are used, even with extensive hyperparameter tuning. Our findings underscore critical considerations for ML practitioners who pre-train models using large-scale web-curated data and are concerned about potential backdoor threats. Notably, our results suggest that simpler pre-training objectives are more amenable to effective backdoor removal. This insight is pivotal for practitioners seeking to balance the trade-offs between using stronger pre-training objectives and security against backdoor attacks.



## **9. AnonPSI: An Anonymity Assessment Framework for PSI**

cs.CR

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.18118v1) [paper-pdf](http://arxiv.org/pdf/2311.18118v1)

**Authors**: Bo Jiang, Jian Du, Qiang Yan

**Abstract**: Private Set Intersection (PSI) is a widely used protocol that enables two parties to securely compute a function over the intersected part of their shared datasets and has been a significant research focus over the years. However, recent studies have highlighted its vulnerability to Set Membership Inference Attacks (SMIA), where an adversary might deduce an individual's membership by invoking multiple PSI protocols. This presents a considerable risk, even in the most stringent versions of PSI, which only return the cardinality of the intersection. This paper explores the evaluation of anonymity within the PSI context. Initially, we highlight the reasons why existing works fall short in measuring privacy leakage, and subsequently propose two attack strategies that address these deficiencies. Furthermore, we provide theoretical guarantees on the performance of our proposed methods. In addition to these, we illustrate how the integration of auxiliary information, such as the sum of payloads associated with members of the intersection (PSI-SUM), can enhance attack efficiency. We conducted a comprehensive performance evaluation of various attack strategies proposed utilizing two real datasets. Our findings indicate that the methods we propose markedly enhance attack efficiency when contrasted with previous research endeavors. {The effective attacking implies that depending solely on existing PSI protocols may not provide an adequate level of privacy assurance. It is recommended to combine privacy-enhancing technologies synergistically to enhance privacy protection even further.



## **10. Improving Faithfulness for Vision Transformers**

cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17983v1) [paper-pdf](http://arxiv.org/pdf/2311.17983v1)

**Authors**: Lijie Hu, Yixin Liu, Ninghao Liu, Mengdi Huai, Lichao Sun, Di Wang

**Abstract**: Vision Transformers (ViTs) have achieved state-of-the-art performance for various vision tasks. One reason behind the success lies in their ability to provide plausible innate explanations for the behavior of neural architectures. However, ViTs suffer from issues with explanation faithfulness, as their focal points are fragile to adversarial attacks and can be easily changed with even slight perturbations on the input image. In this paper, we propose a rigorous approach to mitigate these issues by introducing Faithful ViTs (FViTs). Briefly speaking, an FViT should have the following two properties: (1) The top-$k$ indices of its self-attention vector should remain mostly unchanged under input perturbation, indicating stable explanations; (2) The prediction distribution should be robust to perturbations. To achieve this, we propose a new method called Denoised Diffusion Smoothing (DDS), which adopts randomized smoothing and diffusion-based denoising. We theoretically prove that processing ViTs directly with DDS can turn them into FViTs. We also show that Gaussian noise is nearly optimal for both $\ell_2$ and $\ell_\infty$-norm cases. Finally, we demonstrate the effectiveness of our approach through comprehensive experiments and evaluations. Specifically, we compare our FViTs with other baselines through visual interpretation and robustness accuracy under adversarial attacks. Results show that FViTs are more robust against adversarial attacks while maintaining the explainability of attention, indicating higher faithfulness.



## **11. On the Adversarial Robustness of Graph Contrastive Learning Methods**

cs.LG

Accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop  (NeurIPS GLFrontiers 2023)

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17853v1) [paper-pdf](http://arxiv.org/pdf/2311.17853v1)

**Authors**: Filippo Guerranti, Zinuo Yi, Anna Starovoit, Rafiq Kamel, Simon Geisler, Stephan Günnemann

**Abstract**: Contrastive learning (CL) has emerged as a powerful framework for learning representations of images and text in a self-supervised manner while enhancing model robustness against adversarial attacks. More recently, researchers have extended the principles of contrastive learning to graph-structured data, giving birth to the field of graph contrastive learning (GCL). However, whether GCL methods can deliver the same advantages in adversarial robustness as their counterparts in the image and text domains remains an open question. In this paper, we introduce a comprehensive robustness evaluation protocol tailored to assess the robustness of GCL models. We subject these models to adaptive adversarial attacks targeting the graph structure, specifically in the evasion scenario. We evaluate node and graph classification tasks using diverse real-world datasets and attack strategies. With our work, we aim to offer insights into the robustness of GCL methods and hope to open avenues for potential future research directions.



## **12. SenTest: Evaluating Robustness of Sentence Encoders**

cs.CL

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17722v1) [paper-pdf](http://arxiv.org/pdf/2311.17722v1)

**Authors**: Tanmay Chavan, Shantanu Patankar, Aditya Kane, Omkar Gokhale, Geetanjali Kale, Raviraj Joshi

**Abstract**: Contrastive learning has proven to be an effective method for pre-training models using weakly labeled data in the vision domain. Sentence transformers are the NLP counterparts to this architecture, and have been growing in popularity due to their rich and effective sentence representations. Having effective sentence representations is paramount in multiple tasks, such as information retrieval, retrieval augmented generation (RAG), and sentence comparison. Keeping in mind the deployability factor of transformers, evaluating the robustness of sentence transformers is of utmost importance. This work focuses on evaluating the robustness of the sentence encoders. We employ several adversarial attacks to evaluate its robustness. This system uses character-level attacks in the form of random character substitution, word-level attacks in the form of synonym replacement, and sentence-level attacks in the form of intra-sentence word order shuffling. The results of the experiments strongly undermine the robustness of sentence encoders. The models produce significantly different predictions as well as embeddings on perturbed datasets. The accuracy of the models can fall up to 15 percent on perturbed datasets as compared to unperturbed datasets. Furthermore, the experiments demonstrate that these embeddings does capture the semantic and syntactic structure (sentence order) of sentences. However, existing supervised classification strategies fail to leverage this information, and merely function as n-gram detectors.



## **13. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

cs.LG

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2310.03684v3) [paper-pdf](http://arxiv.org/pdf/2310.03684v3)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM. Our code is publicly available at the following link: https://github.com/arobey1/smooth-llm.



## **14. Natural & Adversarial Bokeh Rendering via Circle-of-Confusion Predictive Network**

cs.CV

11 pages, accepted by TMM

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2111.12971v3) [paper-pdf](http://arxiv.org/pdf/2111.12971v3)

**Authors**: Yihao Huang, Felix Juefei-Xu, Qing Guo, Geguang Pu, Yang Liu

**Abstract**: Bokeh effect is a natural shallow depth-of-field phenomenon that blurs the out-of-focus part in photography. In recent years, a series of works have proposed automatic and realistic bokeh rendering methods for artistic and aesthetic purposes. They usually employ cutting-edge data-driven deep generative networks with complex training strategies and network architectures. However, these works neglect that the bokeh effect, as a real phenomenon, can inevitably affect the subsequent visual intelligent tasks like recognition, and their data-driven nature prevents them from studying the influence of bokeh-related physical parameters (i.e., depth-of-the-field) on the intelligent tasks. To fill this gap, we study a totally new problem, i.e., natural & adversarial bokeh rendering, which consists of two objectives: rendering realistic and natural bokeh and fooling the visual perception models (i.e., bokeh-based adversarial attack). To this end, beyond the pure data-driven solution, we propose a hybrid alternative by taking the respective advantages of data-driven and physical-aware methods. Specifically, we propose the circle-of-confusion predictive network (CoCNet) by taking the all-in-focus image and depth image as inputs to estimate circle-of-confusion parameters for each pixel, which are employed to render the final image through a well-known physical model of bokeh. With the hybrid solution, our method could achieve more realistic rendering results with the naive training strategy and a much lighter network.



## **15. Query-Relevant Images Jailbreak Large Multi-Modal Models**

cs.CV

Technique report

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17600v1) [paper-pdf](http://arxiv.org/pdf/2311.17600v1)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Warning: This paper contains examples of harmful language and images, and reader discretion is recommended. The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Large Multi-Modal Models (LMMs) remains understudied. In our study, we present a novel visual prompt attack that exploits query-relevant images to jailbreak the open-source LMMs. Our method creates a composite image from one image generated by diffusion models and another that displays the text as typography, based on keywords extracted from a malicious query. We show LLMs can be easily attacked by our approach, even if the employed Large Language Models are safely aligned. To evaluate the extent of this vulnerability in open-source LMMs, we have compiled a substantial dataset encompassing 13 scenarios with a total of 5,040 text-image pairs, using our presented attack technique. Our evaluation of 12 cutting-edge LMMs using this dataset shows the vulnerability of existing multi-modal models on adversarial attacks. This finding underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source LMMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **16. Quantum Neural Networks under Depolarization Noise: Exploring White-Box Attacks and Defenses**

quant-ph

Poster at Quantum Techniques in Machine Learning (QTML) 2023

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17458v1) [paper-pdf](http://arxiv.org/pdf/2311.17458v1)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: Leveraging the unique properties of quantum mechanics, Quantum Machine Learning (QML) promises computational breakthroughs and enriched perspectives where traditional systems reach their boundaries. However, similarly to classical machine learning, QML is not immune to adversarial attacks. Quantum adversarial machine learning has become instrumental in highlighting the weak points of QML models when faced with adversarial crafted feature vectors. Diving deep into this domain, our exploration shines light on the interplay between depolarization noise and adversarial robustness. While previous results enhanced robustness from adversarial threats through depolarization noise, our findings paint a different picture. Interestingly, adding depolarization noise discontinued the effect of providing further robustness for a multi-class classification scenario. Consolidating our findings, we conducted experiments with a multi-class classifier adversarially trained on gate-based quantum simulators, further elucidating this unexpected behavior.



## **17. Group-wise Sparse and Explainable Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17434v1) [paper-pdf](http://arxiv.org/pdf/2311.17434v1)

**Authors**: Shpresim Sadiku, Moritz Wagner, Sebastian Pokutta

**Abstract**: Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, typically regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs than previously anticipated. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. In this paper, we tackle this challenge by presenting an algorithm that simultaneously generates group-wise sparse attacks within semantically meaningful areas of an image. In each iteration, the core operation of our algorithm involves the optimization of a quasinorm adversarial loss. This optimization is achieved by employing the $1/2$-quasinorm proximal operator for some iterations, a method tailored for nonconvex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2$-norm regularization applied to perturbation magnitudes. We rigorously evaluate the efficacy of our novel attack in both targeted and non-targeted attack scenarios, on CIFAR-10 and ImageNet datasets. When compared to state-of-the-art methods, our attack consistently results in a remarkable increase in group-wise sparsity, e.g., an increase of $48.12\%$ on CIFAR-10 and $40.78\%$ on ImageNet (average case, targeted attack), all while maintaining lower perturbation magnitudes. Notably, this performance is complemented by a significantly faster computation time and a $100\%$ attack success rate.



## **18. Enhancing Adversarial Attacks: The Similar Target Method**

cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2308.10743v3) [paper-pdf](http://arxiv.org/pdf/2308.10743v3)

**Authors**: Shuo Zhang, Ziruo Wang, Zikai Zhou, Huanran Chen

**Abstract**: Deep neural networks are vulnerable to adversarial examples, posing a threat to the models' applications and raising security concerns. An intriguing property of adversarial examples is their strong transferability. Several methods have been proposed to enhance transferability, including ensemble attacks which have demonstrated their efficacy. However, prior approaches simply average logits, probabilities, or losses for model ensembling, lacking a comprehensive analysis of how and why model ensembling significantly improves transferability. In this paper, we propose a similar targeted attack method named Similar Target~(ST). By promoting cosine similarity between the gradients of each model, our method regularizes the optimization direction to simultaneously attack all surrogate models. This strategy has been proven to enhance generalization ability. Experimental results on ImageNet validate the effectiveness of our approach in improving adversarial transferability. Our method outperforms state-of-the-art attackers on 18 discriminative classifiers and adversarially trained models.



## **19. RADAP: A Robust and Adaptive Defense Against Diverse Adversarial Patches on Face Recognition**

cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17339v1) [paper-pdf](http://arxiv.org/pdf/2311.17339v1)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao, Changhai Nie

**Abstract**: Face recognition (FR) systems powered by deep learning have become widely used in various applications. However, they are vulnerable to adversarial attacks, especially those based on local adversarial patches that can be physically applied to real-world objects. In this paper, we propose RADAP, a robust and adaptive defense mechanism against diverse adversarial patches in both closed-set and open-set FR systems. RADAP employs innovative techniques, such as FCutout and F-patch, which use Fourier space sampling masks to improve the occlusion robustness of the FR model and the performance of the patch segmenter. Moreover, we introduce an edge-aware binary cross-entropy (EBCE) loss function to enhance the accuracy of patch detection. We also present the split and fill (SAF) strategy, which is designed to counter the vulnerability of the patch segmenter to complete white-box adaptive attacks. We conduct comprehensive experiments to validate the effectiveness of RADAP, which shows significant improvements in defense performance against various adversarial patches, while maintaining clean accuracy higher than that of the undefended Vanilla model.



## **20. NeRFTAP: Enhancing Transferability of Adversarial Patches on Face Recognition using Neural Radiance Fields**

cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17332v1) [paper-pdf](http://arxiv.org/pdf/2311.17332v1)

**Authors**: Xiaoliang Liu, Furao Shen, Feng Han, Jian Zhao, Changhai Nie

**Abstract**: Face recognition (FR) technology plays a crucial role in various applications, but its vulnerability to adversarial attacks poses significant security concerns. Existing research primarily focuses on transferability to different FR models, overlooking the direct transferability to victim's face images, which is a practical threat in real-world scenarios. In this study, we propose a novel adversarial attack method that considers both the transferability to the FR model and the victim's face image, called NeRFTAP. Leveraging NeRF-based 3D-GAN, we generate new view face images for the source and target subjects to enhance transferability of adversarial patches. We introduce a style consistency loss to ensure the visual similarity between the adversarial UV map and the target UV map under a 0-1 mask, enhancing the effectiveness and naturalness of the generated adversarial face images. Extensive experiments and evaluations on various FR models demonstrate the superiority of our approach over existing attack techniques. Our work provides valuable insights for enhancing the robustness of FR systems in practical adversarial settings.



## **21. Content-based Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2305.10665v2) [paper-pdf](http://arxiv.org/pdf/2305.10665v2)

**Authors**: Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic, demonstrating their ability to deceive human perception and deep neural networks with stealth and success. However, current works usually sacrifice unrestricted degrees and subjectively select some image content to guarantee the photorealism of unrestricted adversarial examples, which limits its attack performance. To ensure the photorealism of adversarial examples and boost attack performance, we propose a novel unrestricted attack framework called Content-based Unrestricted Adversarial Attack. By leveraging a low-dimensional manifold that represents natural images, we map the images onto the manifold and optimize them along its adversarial direction. Therefore, within this framework, we implement Adversarial Content Attack based on Stable Diffusion and can generate high transferable unrestricted adversarial examples with various adversarial contents. Extensive experimentation and visualization demonstrate the efficacy of ACA, particularly in surpassing state-of-the-art attacks by an average of 13.3-50.4% and 16.8-48.0% in normally trained models and defense methods, respectively.



## **22. Advancing Attack-Resilient Scheduling of Integrated Energy Systems with Demand Response via Deep Reinforcement Learning**

eess.SY

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17941v1) [paper-pdf](http://arxiv.org/pdf/2311.17941v1)

**Authors**: Yang Li, Wenjie Ma, Yuanzheng Li, Sen Li, Zhe Chen

**Abstract**: Optimally scheduling multi-energy flow is an effective method to utilize renewable energy sources (RES) and improve the stability and economy of integrated energy systems (IES). However, the stable demand-supply of IES faces challenges from uncertainties that arise from RES and loads, as well as the increasing impact of cyber-attacks with advanced information and communication technologies adoption. To address these challenges, this paper proposes an innovative model-free resilience scheduling method based on state-adversarial deep reinforcement learning (DRL) for integrated demand response (IDR)-enabled IES. The proposed method designs an IDR program to explore the interaction ability of electricity-gas-heat flexible loads. Additionally, a state-adversarial Markov decision process (SA-MDP) model characterizes the energy scheduling problem of IES under cyber-attack. The state-adversarial soft actor-critic (SA-SAC) algorithm is proposed to mitigate the impact of cyber-attacks on the scheduling strategy. Simulation results demonstrate that our method is capable of adequately addressing the uncertainties resulting from RES and loads, mitigating the impact of cyber-attacks on the scheduling strategy, and ensuring a stable demand supply for various energy sources. Moreover, the proposed method demonstrates resilience against cyber-attacks. Compared to the original soft actor-critic (SAC) algorithm, it achieves a 10\% improvement in economic performance under cyber-attack scenarios.



## **23. Scalable Extraction of Training Data from (Production) Language Models**

cs.LG

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17035v1) [paper-pdf](http://arxiv.org/pdf/2311.17035v1)

**Authors**: Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, Katherine Lee

**Abstract**: This paper studies extractable memorization: training data that an adversary can efficiently extract by querying a machine learning model without prior knowledge of the training dataset. We show an adversary can extract gigabytes of training data from open-source language models like Pythia or GPT-Neo, semi-open models like LLaMA or Falcon, and closed models like ChatGPT. Existing techniques from the literature suffice to attack unaligned models; in order to attack the aligned ChatGPT, we develop a new divergence attack that causes the model to diverge from its chatbot-style generations and emit training data at a rate 150x higher than when behaving properly. Our methods show practical attacks can recover far more data than previously thought, and reveal that current alignment techniques do not eliminate memorization.



## **24. Breaking Boundaries: Balancing Performance and Robustness in Deep Wireless Traffic Forecasting**

cs.LG

Accepted for presentation at the ARTMAN workshop, part of the ACM  Conference on Computer and Communications Security (CCS), 2023

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.09790v3) [paper-pdf](http://arxiv.org/pdf/2311.09790v3)

**Authors**: Romain Ilbert, Thai V. Hoang, Zonghua Zhang, Themis Palpanas

**Abstract**: Balancing the trade-off between accuracy and robustness is a long-standing challenge in time series forecasting. While most of existing robust algorithms have achieved certain suboptimal performance on clean data, sustaining the same performance level in the presence of data perturbations remains extremely hard. In this paper, we study a wide array of perturbation scenarios and propose novel defense mechanisms against adversarial attacks using real-world telecom data. We compare our strategy against two existing adversarial training algorithms under a range of maximal allowed perturbations, defined using $\ell_{\infty}$-norm, $\in [0.1,0.4]$. Our findings reveal that our hybrid strategy, which is composed of a classifier to detect adversarial examples, a denoiser to eliminate noise from the perturbed data samples, and a standard forecaster, achieves the best performance on both clean and perturbed data. Our optimal model can retain up to $92.02\%$ the performance of the original forecasting model in terms of Mean Squared Error (MSE) on clean data, while being more robust than the standard adversarially trained models on perturbed data. Its MSE is 2.71$\times$ and 2.51$\times$ lower than those of comparing methods on normal and perturbed data, respectively. In addition, the components of our models can be trained in parallel, resulting in better computational efficiency. Our results indicate that we can optimally balance the trade-off between the performance and robustness of forecasting models by improving the classifier and denoiser, even in the presence of sophisticated and destructive poisoning attacks.



## **25. Vulnerability Analysis of Transformer-based Optical Character Recognition to Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17128v1) [paper-pdf](http://arxiv.org/pdf/2311.17128v1)

**Authors**: Lucas Beerens, Desmond J. Higham

**Abstract**: Recent advancements in Optical Character Recognition (OCR) have been driven by transformer-based models. OCR systems are critical in numerous high-stakes domains, yet their vulnerability to adversarial attack remains largely uncharted territory, raising concerns about security and compliance with emerging AI regulations. In this work we present a novel framework to assess the resilience of Transformer-based OCR (TrOCR) models. We develop and assess algorithms for both targeted and untargeted attacks. For the untargeted case, we measure the Character Error Rate (CER), while for the targeted case we use the success ratio. We find that TrOCR is highly vulnerable to untargeted attacks and somewhat less vulnerable to targeted attacks. On a benchmark handwriting data set, untargeted attacks can cause a CER of more than 1 without being noticeable to the eye. With a similar perturbation size, targeted attacks can lead to success rates of around $25\%$ -- here we attacked single tokens, requiring TrOCR to output the tenth most likely token from a large vocabulary.



## **26. Generation of Games for Opponent Model Differentiation**

cs.AI

4 pages

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16781v1) [paper-pdf](http://arxiv.org/pdf/2311.16781v1)

**Authors**: David Milec, Viliam Lisý, Christopher Kiekintveld

**Abstract**: Protecting against adversarial attacks is a common multiagent problem. Attackers in the real world are predominantly human actors, and the protection methods often incorporate opponent models to improve the performance when facing humans. Previous results show that modeling human behavior can significantly improve the performance of the algorithms. However, modeling humans correctly is a complex problem, and the models are often simplified and assume humans make mistakes according to some distribution or train parameters for the whole population from which they sample. In this work, we use data gathered by psychologists who identified personality types that increase the likelihood of performing malicious acts. However, in the previous work, the tests on a handmade game could not show strategic differences between the models. We created a novel model that links its parameters to psychological traits. We optimized over parametrized games and created games in which the differences are profound. Our work can help with automatic game generation when we need a game in which some models will behave differently and to identify situations in which the models do not align.



## **27. Cooperative Abnormal Node Detection with Adversary Resistance: A Probabilistic Approach**

eess.SY

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16661v1) [paper-pdf](http://arxiv.org/pdf/2311.16661v1)

**Authors**: Yingying Huangfu, Tian Bai

**Abstract**: This paper presents a novel probabilistic detection scheme called Cooperative Statistical Detection (CSD) for abnormal node detection while defending against adversarial attacks in cluster-tree networks. The CSD performs a two-phase process: 1) designing a likelihood ratio test (LRT) for a non-root node at its children from the perspective of packet loss; 2) making an overall decision at the root node based on the aggregated detection data of the nodes over tree branches. In most adversarial scenarios, malicious children knowing the detection policy can generate falsified data to protect the abnormal parent from being detected or frame its normal parent as an anomalous node. To resolve this issue, a modified Z-score-based falsification-resistant mechanism is presented in the CSD to remove untrustworthy information. Through theoretical analysis, we show that the LRT-based method achieves perfect detection, i.e., both the false alarm and missed detection probabilities decay exponentially to zero. Furthermore, the optimal removal threshold of the modified Z-score method is derived for falsifications with uncertain strategies and guarantees perfect detection of the CSD. As our simulation results show, the CSD approach is robust to falsifications and can rapidly reach $99\%$ detection accuracy, even in existing adversarial scenarios, which outperforms state-of-the-art technology.



## **28. On the Role of Randomization in Adversarially Robust Classification**

cs.LG

10 pages main paper (27 total), 2 figures in main paper. Neurips 2023

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2302.07221v3) [paper-pdf](http://arxiv.org/pdf/2302.07221v3)

**Authors**: Lucas Gnecco-Heredia, Yann Chevaleyre, Benjamin Negrevergne, Laurent Meunier, Muni Sreenivas Pydi

**Abstract**: Deep neural networks are known to be vulnerable to small adversarial perturbations in test data. To defend against adversarial attacks, probabilistic classifiers have been proposed as an alternative to deterministic ones. However, literature has conflicting findings on the effectiveness of probabilistic classifiers in comparison to deterministic ones. In this paper, we clarify the role of randomization in building adversarially robust classifiers. Given a base hypothesis set of deterministic classifiers, we show the conditions under which a randomized ensemble outperforms the hypothesis set in adversarial risk, extending previous results. Additionally, we show that for any probabilistic binary classifier (including randomized ensembles), there exists a deterministic classifier that outperforms it. Finally, we give an explicit description of the deterministic hypothesis set that contains such a deterministic classifier for many types of commonly used probabilistic classifiers, i.e. randomized ensembles and parametric/input noise injection.



## **29. Efficient Key-Based Adversarial Defense for ImageNet by Using Pre-trained Model**

cs.CV

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16577v1) [paper-pdf](http://arxiv.org/pdf/2311.16577v1)

**Authors**: AprilPyone MaungMaung, Isao Echizen, Hitoshi Kiya

**Abstract**: In this paper, we propose key-based defense model proliferation by leveraging pre-trained models and utilizing recent efficient fine-tuning techniques on ImageNet-1k classification. First, we stress that deploying key-based models on edge devices is feasible with the latest model deployment advancements, such as Apple CoreML, although the mainstream enterprise edge artificial intelligence (Edge AI) has been focused on the Cloud. Then, we point out that the previous key-based defense on on-device image classification is impractical for two reasons: (1) training many classifiers from scratch is not feasible, and (2) key-based defenses still need to be thoroughly tested on large datasets like ImageNet. To this end, we propose to leverage pre-trained models and utilize efficient fine-tuning techniques to proliferate key-based models even on limited computing resources. Experiments were carried out on the ImageNet-1k dataset using adaptive and non-adaptive attacks. The results show that our proposed fine-tuned key-based models achieve a superior classification accuracy (more than 10% increase) compared to the previous key-based models on classifying clean and adversarial examples.



## **30. Adversarial Doodles: Interpretable and Human-drawable Attacks Provide Describable Insights**

cs.CV

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.15994v2) [paper-pdf](http://arxiv.org/pdf/2311.15994v2)

**Authors**: Ryoya Nara, Yusuke Matsui

**Abstract**: DNN-based image classification models are susceptible to adversarial attacks. Most previous adversarial attacks do not focus on the interpretability of the generated adversarial examples, and we cannot gain insights into the mechanism of the target classifier from the attacks. Therefore, we propose Adversarial Doodles, which have interpretable shapes. We optimize black b\'ezier curves to fool the target classifier by overlaying them onto the input image. By introducing random perspective transformation and regularizing the doodled area, we obtain compact attacks that cause misclassification even when humans replicate them by hand. Adversarial doodles provide describable and intriguing insights into the relationship between our attacks and the classifier's output. We utilize adversarial doodles and discover the bias inherent in the target classifier, such as "We add two strokes on its head, a triangle onto its body, and two lines inside the triangle on a bird image. Then, the classifier misclassifies the image as a butterfly."



## **31. Threshold Breaker: Can Counter-Based RowHammer Prevention Mechanisms Truly Safeguard DRAM?**

cs.AR

7 pages, 6 figures

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16460v1) [paper-pdf](http://arxiv.org/pdf/2311.16460v1)

**Authors**: Ranyang Zhou, Jacqueline Liu, Sabbir Ahmed, Nakul Kochar, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: This paper challenges the existing victim-focused counter-based RowHammer detection mechanisms by experimentally demonstrating a novel multi-sided fault injection attack technique called Threshold Breaker. This mechanism can effectively bypass the most advanced counter-based defense mechanisms by soft-attacking the rows at a farther physical distance from the target rows. While no prior work has demonstrated the effect of such an attack, our work closes this gap by systematically testing 128 real commercial DDR4 DRAM products and reveals that the Threshold Breaker affects various chips from major DRAM manufacturers. As a case study, we compare the performance efficiency between our mechanism and a well-known double-sided attack by performing adversarial weight attacks on a modern Deep Neural Network (DNN). The results demonstrate that the Threshold Breaker can deliberately deplete the intelligence of the targeted DNN system while DRAM is fully protected.



## **32. Rethinking Mixup for Improving the Adversarial Transferability**

cs.CV

13 pages, 8 figures, 4 tables

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17087v1) [paper-pdf](http://arxiv.org/pdf/2311.17087v1)

**Authors**: Xiaosen Wang, Zeyuan Yin

**Abstract**: Mixup augmentation has been widely integrated to generate adversarial examples with superior adversarial transferability when immigrating from a surrogate model to other models. However, the underlying mechanism influencing the mixup's effect on transferability remains unexplored. In this work, we posit that the adversarial examples located at the convergence of decision boundaries across various categories exhibit better transferability and identify that Admix tends to steer the adversarial examples towards such regions. However, we find the constraint on the added image in Admix decays its capability, resulting in limited transferability. To address such an issue, we propose a new input transformation-based attack called Mixing the Image but Separating the gradienT (MIST). Specifically, MIST randomly mixes the input image with a randomly shifted image and separates the gradient of each loss item for each mixed image. To counteract the imprecise gradient, MIST calculates the gradient on several mixed images for each input sample. Extensive experimental results on the ImageNet dataset demonstrate that MIST outperforms existing SOTA input transformation-based attacks with a clear margin on both Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) w/wo defense mechanisms, supporting MIST's high effectiveness and generality.



## **33. Certifying LLM Safety against Adversarial Prompting**

cs.CL

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2309.02705v2) [paper-pdf](http://arxiv.org/pdf/2309.02705v2)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial attacks, which add maliciously designed token sequences to a harmful prompt to bypass the model's safety guards. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 92% of harmful prompts and labels 94% of safe prompts correctly using the open-source language model Llama 2 as the safety filter. We further improve the filter's performance, in terms of accuracy and speed, by replacing Llama 2 with a DistilBERT safety classifier fine-tuned on safe and harmful prompts. Additionally, we propose two efficient empirical defenses: i) RandEC, a randomized version of erase-and-check that evaluates the safety filter on a small subset of the erased subsequences, and ii) GradEC, a gradient-based version that optimizes the erased tokens to remove the adversarial sequence. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.



## **34. Mate! Are You Really Aware? An Explainability-Guided Testing Framework for Robustness of Malware Detectors**

cs.CR

Accepted at ESEC/FSE 2023. https://doi.org/10.1145/3611643.3616309

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2111.10085v4) [paper-pdf](http://arxiv.org/pdf/2111.10085v4)

**Authors**: Ruoxi Sun, Minhui Xue, Gareth Tyson, Tian Dong, Shaofeng Li, Shuo Wang, Haojin Zhu, Seyit Camtepe, Surya Nepal

**Abstract**: Numerous open-source and commercial malware detectors are available. However, their efficacy is threatened by new adversarial attacks, whereby malware attempts to evade detection, e.g., by performing feature-space manipulation. In this work, we propose an explainability-guided and model-agnostic testing framework for robustness of malware detectors when confronted with adversarial attacks. The framework introduces the concept of Accrued Malicious Magnitude (AMM) to identify which malware features could be manipulated to maximize the likelihood of evading detection. We then use this framework to test several state-of-the-art malware detectors' abilities to detect manipulated malware. We find that (i) commercial antivirus engines are vulnerable to AMM-guided test cases; (ii) the ability of a manipulated malware generated using one detector to evade detection by another detector (i.e., transferability) depends on the overlap of features with large AMM values between the different detectors; and (iii) AMM values effectively measure the fragility of features (i.e., capability of feature-space manipulation to flip the prediction results) and explain the robustness of malware detectors facing evasion attacks. Our findings shed light on the limitations of current malware detectors, as well as how they can be improved.



## **35. How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs**

cs.CV

H.T., C.C., and Z.W. contribute equally. Work done during H.T. and  Z.W.'s internship at UCSC, and C.C. and Y.Z.'s internship at UNC

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.16101v1) [paper-pdf](http://arxiv.org/pdf/2311.16101v1)

**Authors**: Haoqin Tu, Chenhang Cui, Zijun Wang, Yiyang Zhou, Bingchen Zhao, Junlin Han, Wangchunshu Zhou, Huaxiu Yao, Cihang Xie

**Abstract**: This work focuses on the potential of Vision LLMs (VLLMs) in visual reasoning. Different from prior studies, we shift our focus from evaluating standard performance to introducing a comprehensive safety evaluation suite, covering both out-of-distribution (OOD) generalization and adversarial robustness. For the OOD evaluation, we present two novel VQA datasets, each with one variant, designed to test model performance under challenging conditions. In exploring adversarial robustness, we propose a straightforward attack strategy for misleading VLLMs to produce visual-unrelated responses. Moreover, we assess the efficacy of two jailbreaking strategies, targeting either the vision or language component of VLLMs. Our evaluation of 21 diverse models, ranging from open-source VLLMs to GPT-4V, yields interesting observations: 1) Current VLLMs struggle with OOD texts but not images, unless the visual information is limited; and 2) These VLLMs can be easily misled by deceiving vision encoders only, and their vision-language training often compromise safety protocols. We release this safety evaluation suite at https://github.com/UCSC-VLAA/vllm-safety-benchmark.



## **36. CALICO: Self-Supervised Camera-LiDAR Contrastive Pre-training for BEV Perception**

cs.CV

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2306.00349v2) [paper-pdf](http://arxiv.org/pdf/2306.00349v2)

**Authors**: Jiachen Sun, Haizhong Zheng, Qingzhao Zhang, Atul Prakash, Z. Morley Mao, Chaowei Xiao

**Abstract**: Perception is crucial in the realm of autonomous driving systems, where bird's eye view (BEV)-based architectures have recently reached state-of-the-art performance. The desirability of self-supervised representation learning stems from the expensive and laborious process of annotating 2D and 3D data. Although previous research has investigated pretraining methods for both LiDAR and camera-based 3D object detection, a unified pretraining framework for multimodal BEV perception is missing. In this study, we introduce CALICO, a novel framework that applies contrastive objectives to both LiDAR and camera backbones. Specifically, CALICO incorporates two stages: point-region contrast (PRC) and region-aware distillation (RAD). PRC better balances the region- and scene-level representation learning on the LiDAR modality and offers significant performance improvement compared to existing methods. RAD effectively achieves contrastive distillation on our self-trained teacher model. CALICO's efficacy is substantiated by extensive evaluations on 3D object detection and BEV map segmentation tasks, where it delivers significant performance improvements. Notably, CALICO outperforms the baseline method by 10.5% and 8.6% on NDS and mAP. Moreover, CALICO boosts the robustness of multimodal 3D object detection against adversarial attacks and corruption. Additionally, our framework can be tailored to different backbones and heads, positioning it as a promising approach for multimodal BEV perception.



## **37. AdaptGuard: Defending Against Universal Attacks for Model Adaptation**

cs.CR

ICCV2023

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2303.10594v2) [paper-pdf](http://arxiv.org/pdf/2303.10594v2)

**Authors**: Lijun Sheng, Jian Liang, Ran He, Zilei Wang, Tieniu Tan

**Abstract**: Model adaptation aims at solving the domain transfer problem under the constraint of only accessing the pretrained source models. With the increasing considerations of data privacy and transmission efficiency, this paradigm has been gaining recent popularity. This paper studies the vulnerability to universal attacks transferred from the source domain during model adaptation algorithms due to the existence of malicious providers. We explore both universal adversarial perturbations and backdoor attacks as loopholes on the source side and discover that they still survive in the target models after adaptation. To address this issue, we propose a model preprocessing framework, named AdaptGuard, to improve the security of model adaptation algorithms. AdaptGuard avoids direct use of the risky source parameters through knowledge distillation and utilizes the pseudo adversarial samples under adjusted radius to enhance the robustness. AdaptGuard is a plug-and-play module that requires neither robust pretrained models nor any changes for the following model adaptation algorithms. Extensive results on three commonly used datasets and two popular adaptation methods validate that AdaptGuard can effectively defend against universal attacks and maintain clean accuracy in the target domain simultaneously. We hope this research will shed light on the safety and robustness of transfer learning. Code is available at https://github.com/TomSheng21/AdaptGuard.



## **38. Attend Who is Weak: Enhancing Graph Condensation via Cross-Free Adversarial Training**

cs.LG

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.15772v1) [paper-pdf](http://arxiv.org/pdf/2311.15772v1)

**Authors**: Xinglin Li, Kun Wang, Hanhui Deng, Yuxuan Liang, Di Wu

**Abstract**: In this paper, we study the \textit{graph condensation} problem by compressing the large, complex graph into a concise, synthetic representation that preserves the most essential and discriminative information of structure and features. We seminally propose the concept of Shock Absorber (a type of perturbation) that enhances the robustness and stability of the original graphs against changes in an adversarial training fashion. Concretely, (I) we forcibly match the gradients between pre-selected graph neural networks (GNNs) trained on a synthetic, simplified graph and the original training graph at regularly spaced intervals. (II) Before each update synthetic graph point, a Shock Absorber serves as a gradient attacker to maximize the distance between the synthetic dataset and the original graph by selectively perturbing the parts that are underrepresented or insufficiently informative. We iteratively repeat the above two processes (I and II) in an adversarial training fashion to maintain the highly-informative context without losing correlation with the original dataset. More importantly, our shock absorber and the synthesized graph parallelly share the backward process in a free training manner. Compared to the original adversarial training, it introduces almost no additional time overhead.   We validate our framework across 8 datasets (3 graph and 5 node classification datasets) and achieve prominent results: for example, on Cora, Citeseer and Ogbn-Arxiv, we can gain nearly 1.13% to 5.03% improvements compare with SOTA models. Moreover, our algorithm adds only about 0.2% to 2.2% additional time overhead over Flicker, Citeseer and Ogbn-Arxiv. Compared to the general adversarial training, our approach improves time efficiency by nearly 4-fold.



## **39. The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing**

cs.LG

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2309.16883v2) [paper-pdf](http://arxiv.org/pdf/2309.16883v2)

**Authors**: Blaise Delattre, Alexandre Araujo, Quentin Barthélemy, Alexandre Allauzen

**Abstract**: Real-life applications of deep neural networks are hindered by their unsteady predictions when faced with noisy inputs and adversarial attacks. The certified radius is in this context a crucial indicator of the robustness of models. However how to design an efficient classifier with a sufficient certified radius? Randomized smoothing provides a promising framework by relying on noise injection in inputs to obtain a smoothed and more robust classifier. In this paper, we first show that the variance introduced by randomized smoothing closely interacts with two other important properties of the classifier, \textit{i.e.} its Lipschitz constant and margin. More precisely, our work emphasizes the dual impact of the Lipschitz constant of the base classifier, on both the smoothed classifier and the empirical variance. Moreover, to increase the certified robust radius, we introduce a different simplex projection technique for the base classifier to leverage the variance-margin trade-off thanks to Bernstein's concentration inequality, along with an enhanced Lipschitz bound. Experimental results show a significant improvement in certified accuracy compared to current state-of-the-art methods. Our novel certification procedure allows us to use pre-trained models that are used with randomized smoothing, effectively improving the current certification radius in a zero-shot manner.



## **40. SLMIA-SR: Speaker-Level Membership Inference Attacks against Speaker Recognition Systems**

cs.CR

In Proceedings of the 31st Network and Distributed System Security  (NDSS) Symposium, 2024

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2309.07983v2) [paper-pdf](http://arxiv.org/pdf/2309.07983v2)

**Authors**: Guangke Chen, Yedi Zhang, Fu Song

**Abstract**: Membership inference attacks allow adversaries to determine whether a particular example was contained in the model's training dataset. While previous works have confirmed the feasibility of such attacks in various applications, none has focused on speaker recognition (SR), a promising voice-based biometric recognition technique. In this work, we propose SLMIA-SR, the first membership inference attack tailored to SR. In contrast to conventional example-level attack, our attack features speaker-level membership inference, i.e., determining if any voices of a given speaker, either the same as or different from the given inference voices, have been involved in the training of a model. It is particularly useful and practical since the training and inference voices are usually distinct, and it is also meaningful considering the open-set nature of SR, namely, the recognition speakers were often not present in the training data. We utilize intra-similarity and inter-dissimilarity, two training objectives of SR, to characterize the differences between training and non-training speakers and quantify them with two groups of features driven by carefully-established feature engineering to mount the attack. To improve the generalizability of our attack, we propose a novel mixing ratio training strategy to train attack models. To enhance the attack performance, we introduce voice chunk splitting to cope with the limited number of inference voices and propose to train attack models dependent on the number of inference voices. Our attack is versatile and can work in both white-box and black-box scenarios. Additionally, we propose two novel techniques to reduce the number of black-box queries while maintaining the attack performance. Extensive experiments demonstrate the effectiveness of SLMIA-SR.



## **41. RetouchUAA: Unconstrained Adversarial Attack via Image Retouching**

cs.CV

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.16478v1) [paper-pdf](http://arxiv.org/pdf/2311.16478v1)

**Authors**: Mengda Xie, Yiling He, Meie Fang

**Abstract**: Deep Neural Networks (DNNs) are susceptible to adversarial examples. Conventional attacks generate controlled noise-like perturbations that fail to reflect real-world scenarios and hard to interpretable. In contrast, recent unconstrained attacks mimic natural image transformations occurring in the real world for perceptible but inconspicuous attacks, yet compromise realism due to neglect of image post-processing and uncontrolled attack direction. In this paper, we propose RetouchUAA, an unconstrained attack that exploits a real-life perturbation: image retouching styles, highlighting its potential threat to DNNs. Compared to existing attacks, RetouchUAA offers several notable advantages. Firstly, RetouchUAA excels in generating interpretable and realistic perturbations through two key designs: the image retouching attack framework and the retouching style guidance module. The former custom-designed human-interpretability retouching framework for adversarial attack by linearizing images while modelling the local processing and retouching decision-making in human retouching behaviour, provides an explicit and reasonable pipeline for understanding the robustness of DNNs against retouching. The latter guides the adversarial image towards standard retouching styles, thereby ensuring its realism. Secondly, attributed to the design of the retouching decision regularization and the persistent attack strategy, RetouchUAA also exhibits outstanding attack capability and defense robustness, posing a heavy threat to DNNs. Experiments on ImageNet and Place365 reveal that RetouchUAA achieves nearly 100\% white-box attack success against three DNNs, while achieving a better trade-off between image naturalness, transferability and defense robustness than baseline attacks.



## **42. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

cs.CL

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.11509v2) [paper-pdf](http://arxiv.org/pdf/2311.11509v2)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that lead to undesirable outputs. The inherent vulnerability of LLMs stems from their input-output mechanisms, especially when presented with intensely out-of-distribution (OOD) inputs. This paper proposes a token-level detection method to identify adversarial prompts, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity and incorporate neighboring token information to encourage the detection of contiguous adversarial prompt sequences. As a result, we propose two methods: one that identifies each token as either being part of an adversarial prompt or not, and another that estimates the probability of each token being part of an adversarial prompt.



## **43. Instruct2Attack: Language-Guided Semantic Adversarial Attacks**

cs.CV

under submission, code coming soon

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.15551v1) [paper-pdf](http://arxiv.org/pdf/2311.15551v1)

**Authors**: Jiang Liu, Chen Wei, Yuxiang Guo, Heng Yu, Alan Yuille, Soheil Feizi, Chun Pong Lau, Rama Chellappa

**Abstract**: We propose Instruct2Attack (I2A), a language-guided semantic attack that generates semantically meaningful perturbations according to free-form language instructions. We make use of state-of-the-art latent diffusion models, where we adversarially guide the reverse diffusion process to search for an adversarial latent code conditioned on the input image and text instruction. Compared to existing noise-based and semantic attacks, I2A generates more natural and diverse adversarial examples while providing better controllability and interpretability. We further automate the attack process with GPT-4 to generate diverse image-specific text instructions. We show that I2A can successfully break state-of-the-art deep neural networks even under strong adversarial defenses, and demonstrate great transferability among a variety of network architectures.



## **44. Confidence Is All You Need for MI Attacks**

cs.LG

2 pages, 1 figure

**SubmitDate**: 2023-11-26    [abs](http://arxiv.org/abs/2311.15373v1) [paper-pdf](http://arxiv.org/pdf/2311.15373v1)

**Authors**: Abhishek Sinha, Himanshi Tibrewal, Mansi Gupta, Nikhar Waghela, Shivank Garg

**Abstract**: In this evolving era of machine learning security, membership inference attacks have emerged as a potent threat to the confidentiality of sensitive data. In this attack, adversaries aim to determine whether a particular point was used during the training of a target model. This paper proposes a new method to gauge a data point's membership in a model's training set. Instead of correlating loss with membership, as is traditionally done, we have leveraged the fact that training examples generally exhibit higher confidence values when classified into their actual class. During training, the model is essentially being 'fit' to the training data and might face particular difficulties in generalization to unseen data. This asymmetry leads to the model achieving higher confidence on the training data as it exploits the specific patterns and noise present in the training data. Our proposed approach leverages the confidence values generated by the machine learning model. These confidence values provide a probabilistic measure of the model's certainty in its predictions and can further be used to infer the membership of a given data point. Additionally, we also introduce another variant of our method that allows us to carry out this attack without knowing the ground truth(true class) of a given data point, thus offering an edge over existing label-dependent attack methods.



## **45. Adversarial Purification of Information Masking**

cs.CV

**SubmitDate**: 2023-11-26    [abs](http://arxiv.org/abs/2311.15339v1) [paper-pdf](http://arxiv.org/pdf/2311.15339v1)

**Authors**: Sitong Liu, Zhichao Lian, Shuangquan Zhang, Liang Xiao

**Abstract**: Adversarial attacks meticulously generate minuscule, imperceptible perturbations to images to deceive neural networks. Counteracting these, adversarial purification methods seek to transform adversarial input samples into clean output images to defend against adversarial attacks. Nonetheless, extent generative models fail to effectively eliminate adversarial perturbations, yielding less-than-ideal purification results. We emphasize the potential threat of residual adversarial perturbations to target models, quantitatively establishing a relationship between perturbation scale and attack capability. Notably, the residual perturbations on the purified image primarily stem from the same-position patch and similar patches of the adversarial sample. We propose a novel adversarial purification approach named Information Mask Purification (IMPure), aims to extensively eliminate adversarial perturbations. To obtain an adversarial sample, we first mask part of the patches information, then reconstruct the patches to resist adversarial perturbations from the patches. We reconstruct all patches in parallel to obtain a cohesive image. Then, in order to protect the purified samples against potential similar regional perturbations, we simulate this risk by randomly mixing the purified samples with the input samples before inputting them into the feature extraction network. Finally, we establish a combined constraint of pixel loss and perceptual loss to augment the model's reconstruction adaptability. Extensive experiments on the ImageNet dataset with three classifier models demonstrate that our approach achieves state-of-the-art results against nine adversarial attack methods. Implementation code and pre-trained weights can be accessed at \textcolor{blue}{https://github.com/NoWindButRain/IMPure}.



## **46. Robust Graph Neural Networks via Unbiased Aggregation**

cs.LG

**SubmitDate**: 2023-11-25    [abs](http://arxiv.org/abs/2311.14934v1) [paper-pdf](http://arxiv.org/pdf/2311.14934v1)

**Authors**: Ruiqi Feng, Zhichao Hou, Tyler Derr, Xiaorui Liu

**Abstract**: The adversarial robustness of Graph Neural Networks (GNNs) has been questioned due to the false sense of security uncovered by strong adaptive attacks despite the existence of numerous defenses. In this work, we delve into the robustness analysis of representative robust GNNs and provide a unified robust estimation point of view to understand their robustness and limitations. Our novel analysis of estimation bias motivates the design of a robust and unbiased graph signal estimator. We then develop an efficient Quasi-Newton iterative reweighted least squares algorithm to solve the estimation problem, which unfolds as robust unbiased aggregation layers in GNNs with a theoretical convergence guarantee. Our comprehensive experiments confirm the strong robustness of our proposed model, and the ablation study provides a deep understanding of its advantages.



## **47. Exploiting Large Language Models (LLMs) through Deception Techniques and Persuasion Principles**

cs.HC

10 pages, 16 tables, 5 figures, IEEE BigData 2023 (Workshops)

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14876v1) [paper-pdf](http://arxiv.org/pdf/2311.14876v1)

**Authors**: Sonali Singh, Faranak Abri, Akbar Siami Namin

**Abstract**: With the recent advent of Large Language Models (LLMs), such as ChatGPT from OpenAI, BARD from Google, Llama2 from Meta, and Claude from Anthropic AI, gain widespread use, ensuring their security and robustness is critical. The widespread use of these language models heavily relies on their reliability and proper usage of this fascinating technology. It is crucial to thoroughly test these models to not only ensure its quality but also possible misuses of such models by potential adversaries for illegal activities such as hacking. This paper presents a novel study focusing on exploitation of such large language models against deceptive interactions. More specifically, the paper leverages widespread and borrows well-known techniques in deception theory to investigate whether these models are susceptible to deceitful interactions.   This research aims not only to highlight these risks but also to pave the way for robust countermeasures that enhance the security and integrity of language models in the face of sophisticated social engineering tactics. Through systematic experiments and analysis, we assess their performance in these critical security domains. Our results demonstrate a significant finding in that these large language models are susceptible to deception and social engineering attacks.



## **48. Adversarial Machine Learning in Latent Representations of Neural Networks**

cs.LG

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2309.17401v2) [paper-pdf](http://arxiv.org/pdf/2309.17401v2)

**Authors**: Milin Zhang, Mohammad Abdi, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have been shown to reduce the computational burden of mobile devices and decrease the end-to-end inference latency in edge computing scenarios. While distributed DNNs have been studied, to the best of our knowledge the resilience of distributed DNNs to adversarial action still remains an open problem. In this paper, we fill the existing research gap by rigorously analyzing the robustness of distributed DNNs against adversarial action. We cast this problem in the context of information theory and introduce two new measurements for distortion and robustness. Our theoretical findings indicate that (i) assuming the same level of information distortion, latent features are always more robust than input representations; (ii) the adversarial robustness is jointly determined by the feature dimension and the generalization capability of the DNN. To test our theoretical findings, we perform extensive experimental analysis by considering 6 different DNN architectures, 6 different approaches for distributed DNN and 10 different adversarial attacks to the ImageNet-1K dataset. Our experimental results support our theoretical findings by showing that the compressed latent representations can reduce the success rate of adversarial attacks by 88% in the best case and by 57% on the average compared to attacks to the input space.



## **49. Tamper-Evident Pairing**

cs.CR

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14790v1) [paper-pdf](http://arxiv.org/pdf/2311.14790v1)

**Authors**: Aleksandar Manev

**Abstract**: Establishing a secure connection between wireless devices has become significantly important with the increasing number of Wi-Fi products coming to the market. In order to provide an easy and secure pairing standard, the Wi-Fi Alliance has designed the Wi-Fi Protected Setup. Push-Button Configuration (PBC) is part of this standard and is especially useful for pairing devices with physical limitations. However, PBC is proven to be vulnerable to man-in-the-middle (MITM) attacks. Tamper-Evident Pairing (TEP) is an improvement of the PBC standard, which aims to fix the MITM vulnerability without interfering the useful properties of PBC. It relies on the Tamper-Evident Announcement (TEA), which guarantees that an adversary can neither tamper a transmitted message without being detected, nor hide the fact that the message has been sent. The security properties of TEP were proven manually by its authors and tested with the Uppaal and Spin model checkers. During the Uppaal model checking, no vulnerabilities were found. However, the Spin model revealed a case, in which the TEP's security is not guaranteed. In this paper, we first provide a comprehensive overview of the TEP protocol, including all information needed to understand how it works. Furthermore, we summarize the security checks performed on it, give the circumstances, under which it is no longer resistant to MITM attacks and explain the reasons why they could not be revealed with the first model. Nevertheless, future work is required to gain full certainty of the TEP's security before applying it in the industry.



## **50. Mind the box: $l_1$-APGD for sparse adversarial attacks on image classifiers**

cs.LG

In ICML 2021. Fixed typos in Eq. (3) and Eq. (4)

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2103.01208v3) [paper-pdf](http://arxiv.org/pdf/2103.01208v3)

**Authors**: Francesco Croce, Matthias Hein

**Abstract**: We show that when taking into account also the image domain $[0,1]^d$, established $l_1$-projected gradient descent (PGD) attacks are suboptimal as they do not consider that the effective threat model is the intersection of the $l_1$-ball and $[0,1]^d$. We study the expected sparsity of the steepest descent step for this effective threat model and show that the exact projection onto this set is computationally feasible and yields better performance. Moreover, we propose an adaptive form of PGD which is highly effective even with a small budget of iterations. Our resulting $l_1$-APGD is a strong white-box attack showing that prior works overestimated their $l_1$-robustness. Using $l_1$-APGD for adversarial training we get a robust classifier with SOTA $l_1$-robustness. Finally, we combine $l_1$-APGD and an adaptation of the Square Attack to $l_1$ into $l_1$-AutoAttack, an ensemble of attacks which reliably assesses adversarial robustness for the threat model of $l_1$-ball intersected with $[0,1]^d$.



