---
title: Session6
layout: single
author: Kerrie Geil
author_profile: true
header:
  overlay_color: "444444"
  overlay_image: /assets/images/margaret-weir-GZyjbLNOaFg-unsplash_dark.jpg
---

<!--NOTE: comment out -->

# Session 6 Symposium:
## Challenges and opportunities in leveraging machine learning techniques to further sustainable and intensified agriculture

<br>
The 2020 SCINet Geospatial Workshop concludes with a symposium showcasing four invited speakers, each giving a 30-minute presentation on how they use machine learning for agricultural research, followed by a panel discussion.


{% capture text %}
The session recording is available for anyone with a usda.gov email address and eAuthentication at (location coming soon).
{% endcapture %}
{% include alert.md text=text %}
<br><br>

## AGENDA	(MDT)
<br>

**11-11:10    Drs Yanghui Kang & Amy Hudson, USDA-ARS SCINet Postdocs**

  - Welcome

**11:10-11:40   [Dr Matthew Jones](#dr-matthew-jones-university-of-montana), University of Montana**

  - Predicting rangeland fractional cover for the western U.S. with random forests and multitask learning

**11:45-12:15   [Dr Liheng Zhong](#dr-liheng-zhong-descartes-labs), Descartes Labs**

  - How to use statistical data to train classifiers

**12:20-12:50   [Dr Vasit Sagan](#dr-vasit-sagan-saint-louis-university), Saint Louis University**

  - UAV-satellite spatio-temporal data fusion and deep learning for yield prediction

**12:55-1:25    [Dr Jingyi Huang](#dr-jingyi-huang-university-of-wisconsin), University of Wisconsin**

  - Characterizing field-scale soil moisture dynamics with big data and machine learning: challenges and opportunities for digital agriculture

**1:30-1:40   Short Break**

**1:40-2:15   Panel Discussion**

- What are the merits and pitfalls of machine learning techniques in comparison to traditional deterministic (physical/process-based) and probabilistic modeling/analysis for agricultural research?
- The USDA ARS is currently undergoing an effort to increase/improve the computational and machine learning capabilities across the agency. What are the skillsets that are required to conduct research with “big data” using machine learning techniques? Do you have suggestions on the best approaches for developing these skillsets?
- What do you see as the future directions for machine learning and advanced computing techniques (e.g. cloud, HPC) in agricultural research?

<br><br>

---

<br>

## Session Rules

**CHAT QUESTIONS/COMMENTS TAKE FIRST PRIORITY** - Chat your question/comments either to everyone (preferred) or to the chat moderator (Kerrie Geil) privately to have your question/comment read out loud anonamously. We will answer chat questions first and call on people who have written in the chat before we take questions from raised hands.

**SHARE YOUR VIDEO WHEN SPEAKING** - If your internet plan/connectivity allows, please share your video when speaking.

**KEEP YOURSELF ON MUTE** - Please mute yourself unless you are called on.
<br><br>

---

<br>

## Invited Speaker Info
<br>

{% capture text %}
### Dr. Matthew Jones, University of Montana
{% endcapture %}
{% include alert.md text=text color=secondary %}

**Title:** Predicting rangeland fractional cover for the western U.S. with random forests and multitask learning

**Abstract:** Capitalizing on over 52,000 on-the-ground vegetation plots we trained two machine learning style models to predict plant functional type fractional cover across western U.S. rangelands, annually from 1984-present. Random Forests provided the initial fractional cover product which we recently improved upon by using multi-task learning in temporal convolutional networks. The pros and cons of the two methods will be discussed as well as the multiple applications of the resulting fractional land cover products.

**Related Publications:**
Allred, B.W., Bestelmeyer, B.T., Boyd, C.S., Brown, C., Davies, K.W., Ellsworth, L.M., Erickson, T.A., Fuhlendorf, S.D., Griffiths, T. V, Jansen, V., **Jones, M.O.**, Karl, J., Maestas, J.D., Maynard, J.J., McCord, S.E., Naugle, D.E., Starns, H.D., Twidwell, D., Uden, D.R., 2020. Improving Landsat predictions of rangeland fractional cover with multitask learning and uncertainty. bioRxiv preprint. [10.1101/2020.06.10.142489](https://www.biorxiv.org/content/10.1101/2020.06.10.142489v1)

**Jones, M. O.**, Allred, B. W., Naugle, D. E., Maestas, J. D., Donnelly, P., Metz, L. J., Karl, J., Smith, R., Bestelmeyer, B., Boyd, C., Kerby, J. D., and McIver, J. D., 2018. Innovation in rangeland monitoring: annual, 30 m, plant functional type percent cover maps for U.S. rangelands, 1984–2017. *Ecosphere* 9( 9):e02430. [10.1002/ecs2.2430](https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.2430)

<br><br>


{% capture text %}
### Dr. Liheng Zhong, Descartes Labs
{% endcapture %}
{% include alert.md text=text color=secondary %}

**Title:** How to use statistical data to train classifiers

**Abstract:** The presentation is about a new possibility brought by deep learning in the field of cropland classification, which is covered in a recent article "Deep learning based winter wheat mapping using statistical data as ground references in Kansas and northern Texas, US" published in Remote Sensing of Environment. In the study, the deep neural network classifier was trained using per-county wheat acreage by USDA and MODIS image series, and the trained classifier predicted per-pixel wheat map without relying on county statistics in the target year.

**Related Publications:**
**Zhong, L.**, Hu, L., Zhou, H., Tao, X., 2019. Deep learning based winter wheat mapping using statistical data as ground references in Kansas and northern Texas, US. *Remote Sens. Environ.* 233, 111411. [10.1016/j.rse.2019.111411](https://doi.org/10.1016/j.rse.2019.111411)

**Zhong, L.**, Hu, L., Zhou, H., 2019. Deep learning based multi-temporal crop classification. *Remote Sens. Environ.* 221, 430–443. [10.1016/j.rse.2018.11.032](https://doi.org/10.1016/j.rse.2018.11.032)

<br><br>


{% capture text %}
### Dr. Vasit Sagan, Saint Louis University
{% endcapture %}
{% include alert.md text=text color=secondary %}

**Title:** UAV-satellite spatio-temporal data fusion and deep learning for yield prediction

**Abstract:** In this work, we present a concept of UAV and satellite spatio-temporal data fusion for crop monitoring, specifically plant phenotyping and yield prediction. We show that (1) spatial-temporal data fusion from airborne and satellite systems provide effective means for capturing early stress; (2) UAV data can complement the limitations of satellite remote sensing data for field-level crop monitoring, addressing not just mixed pixel issues but also filling the temporal gap in satellite data availability; and (3) spatial-temporal gap-filling enables predicting yield more accurately using data collected at optimal growth stages (e.g., seed filling stage). The concept developed in this paper also provides a framework for accurate and robust estimation of plant traits and grain yield and delivers valuable insight for high spatial precision in high-throughput phenotyping and farm field management.

**Related Publications:**
Maimaitijiang, M., **Sagan, V.**, Sidike, P., Hartling, S., Esposito, F., Fritschi, F.B., 2020. Soybean yield prediction from UAV using multimodal data fusion and deep learning. *Remote Sens. Environ.* 237, 111599. [10.1016/j.rse.2019.111599](https://doi.org/10.1016/j.rse.2019.111599)

Maimaitijiang, M.; **Sagan, V.**; Sidike, P.; Daloye, A.M.; Erkbol, H.; Fritschi, F.B., 2020 Crop Monitoring Using Satellite/UAV Data Fusion and Machine Learning. *Remote Sens.* 12, 1357. [10.3390/rs12091357](https://doi.org/10.3390/rs12091357)

<br><br>


{% capture text %}
### Dr. Jingyi Huang, University of Wisconsin
{% endcapture %}
{% include alert.md text=text color=secondary %}

**Title:** Characterizing field-scale soil moisture dynamics with big data and machine learning: challenges and opportunities for digital agriculture

**Abstract:** Knowledge of soil moisture dynamics at the field scale is essential for agricultural management such as irrigation and fertilization scheduling. However, soil moisture varies greatly in space and time. Neither in situ soil moisture sensor networks nor satellite remote sensing platforms can be directly used to guide field-scale agricultural management. Compared to mechanistic and statistical methods, machine learning models have recently shown the potential to leverage big data from remote sensing and in situ soil sensor network measurements and high-resolution land surface parameters for mapping and forecasting soil moisture dynamics at the field, regional, and global scales. Future work is required to improve the machine learning models (e.g. transferability, interpretability), integrate soil moisture forecasts into decision-support systems (e.g. cost-benefit analysis, economic and environmental trade-offs), and enhance cyberinfrastructure (e.g. FAIR principles) for sustainable and intensified agricultural production.

**Related Publications:**
Chatterjee, S., **Huang, J.**, Hartemink, A.E., 2020. Establishing an Empirical Model for Surface Soil Moisture Retrieval at the U.S. Climate Reference Network Using Sentinel-1 Backscatter and Ancillary Data. *Remote Sens.* 12, 1242. [10.3390/rs12081242](https://doi.org/10.3390/rs12081242)

**Huang, J.**, Desai, A.R., Zhu, J., Hartemink, A.E., Stoy, P., II, S.P.L., Zhang, Y., Zhang, Z., Arriaga, F.J., 2020. Retrieving Heterogeneous Surface Soil Moisture at 100 m across the Globe via Synergistic Fusion of Remote Sensing and Land Surface Parameters. *Earth Sp. Sci. Open Arch.* [10.1002/essoar.10502252.1](https://doi.org/10.1002/essoar.10502252.1)
