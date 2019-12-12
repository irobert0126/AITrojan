# AITrojan Detection

Recently, our research has demonstrated a new security threat against AI services. It shows an adversary can disrupt the training pipeline and thus insert Trojan behaviors into the AI systems. This new threat quickly receives the attention from a US government agency -- Intelligence Advanced Research Projects Activity (IARPA). In the summer of 2019, the agency announced a new research program (TrojAI) which seeks for solutions to defend against such trojan attacks. 
 
Technically speaking, trojan attack against neural networks is a practice that injects malicious behaviors into neural networks. For example, an AI learning to distinguish traffic signs can be given potentially just a few additional examples of stop signs with yellow squares on them, each labeled "speed limit sign". If the AI were deployed in a self-driving car, an adversary could cause the car to run through the stop sign just by putting a sticky note on it, since the AI would incorrectly see it as a speed limit sign. 

Recently, trojan attacks have been shown on various types of AI services, ranging from image classification to autonomous vehicles. However, due to the trend that nearly all security companies powered their detection tools by machine learning, we will focus on the impact of trojaning attacks upon these ML-based security products, from malware detection sandbox to firewall, which no one has ever investigated before. In this talk, we will use a live demo to show the consequences of a trojan-implanted security products.

In addition, due to the lack of effective AI trojan detection methods, we will further propose a lightweight, novel approach to defending "trojaning attack" on various machine learning architectures, and will discuss how to adopt such a method to the industry/commercial AI services.

## Detection Result
* Visualization
  * Step 0:
  <img src="Detection/flask/static/result/imgs/adv_step0_all.png" width=888 />

  * Step 500:
  <img src="Detection/flask/static/result/imgs/adv_step500_all.png" width=888 />

  * Step 1000:
  <img src="Detection/flask/static/result/imgs/adv_step1000_all.png" width=888 />

  * Step 1500:
  <img src="Detection/flask/static/result/imgs/adv_step1500_all.png" width=888 />

  * Step 2000:
  <img src="Detection/flask/static/result/imgs/adv_step2000_all.png" width=888 />

  * Step 2500:
  <img src="Detection/flask/static/result/imgs/adv_step2500_all.png" width=888 />
