# Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/cis-lmu/glot500-base)
[![arXiv](https://img.shields.io/badge/arXiv-2305.12182-b31b1b.svg)](https://arxiv.org/abs/2305.12182)

## Introduction
This repository contains information about Glot500, code, model and data.

- [Glot500-m](https://huggingface.co/cis-lmu/glot500-base) is an extended version of [XLM-R](https://huggingface.co/xlm-roberta-base), covering more than **500 languages** compared to XLM-R's 104 languages.

- Glot2000-c comprises corpora for over 2000 languages, while Glot500-c is a subset of Glot2000-c for over 500 languages, including languages with more than 30,000 sentences.



## Glot500-m

You can use this model directly with a pipeline for masked language modeling:

```python
>>> ! pip install transformers
>>> ! pip install sentencepiece
```

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='cis-lmu/glot500-base')
>>> unmasker("Hello I'm a <mask> model.")
```


Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('cis-lmu/glot500-base')
model = AutoModelForMaskedLM.from_pretrained("cis-lmu/glot500-base")

# prepare input
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

# forward pass
output = model(**encoded_input)
```


## Glot500-c

This is an overview of the corpora included Glot500-c presented in our [paper](https://arxiv.org/abs/2305.12182). For requesting access to compiled version of Glot500-c, fill out the [form](https://docs.google.com/forms/d/1FHto_4wWYvEF3lz7DDo3P8wQqfS3WhpYfAu5vM95-qU/viewform?edit_requested=true). Glot500-c will be send via email upon duly completing the form and accepting the license included. 
Please note that, while the data sources utilized in this study do not explicitly prohibit the reuse of data for research purposes, some sources do have copyright statements indicating that such use is permissible, while others do not. Additionally, certain sources prohibit the redistribution of data. As such, data from these sources is omitted from the published version of Glot500-c. 

For more information, check out the table below.

Available |Dataset|Related Papers|Languages |Domain / Notes| Data collection / Verification method|Copyright / Licence|
|:----|:----|:----|:----|:----|:----|:----|
|| AI4Bharat | |  | | 
|| [Akuapem](https://zenodo.org/record/4432117#.Y00gXOxBw-Q)  | - |aka|Parallel sentences|Verified by native speakers|CC-BY-4.0|
|| Anuvaad | |  | | 
|| [Bianet](https://opus.nlpl.eu/Bianet.php) | [Link](http://lrec-conf.org/workshops/lrec2018/W19/pdf/6_W19.pdf) |kur,eng,tur|Parallel news corpus|Automatically curated|CC-BY-SA-4.0 open license|
|| [CMU_Haitian_Creole](http://www.speech.cs.cmu.edu/haitian/text/) | - |hat, eng|Medical domain phrases and sentences in English translated into Haitian Creole by Eriksen Translations, Inc.|Curated|http://www.speech.cs.cmu.edu/haitian/text/COPYING|
|| [FFR](https://github.com/bonaventuredossou/ffr-v1/tree/master/FFR-Dataset) | [Link](https://github.com/bonaventuredossou/ffr-v1/blob/master/FFR-Dataset/FFR_Dataset_Documentation.pdf)|fon, fra|Parallel sentences|Clean curated corpora |MIT License and Licence Creative Commons Attribution - No Commercial Use - Sharing under the Same Conditions 4.0 International.|
|| [GiossaMedia](https://github.com/sgongora27/giossa-gongora-guarani-2021)  |[Link](https://aclanthology.org/2021.americasnlp-1.16.pdf) ; [Link](https://aclanthology.org/2022.computel-1.16.pdf) |spa, grn|Parallel sentences, news and social media|Automatically curated| |
|| [HornMT](https://github.com/asmelashteka/HornMT)  | |aar, amh, eng, orm, som, tir|multi-way parallel corpus| |CC-BY-4.0|
|| [IndicNLP](http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/) | [Link](https://aclanthology.org/2021.wat-1.1/) | <details> ben, guj, hin, kan, mal, mar, tel, tam, ori, pan <summary> expand </summary> </details> |Monolingual and parallel training data|Collection of corpora| |
|| [KinyaSMT](https://github.com/pniyongabo/kinyarwandaSMT)  | [Link](https://github.com/pniyongabo/SeniorThesisPaper/blob/master/finalCopy.pdf) |kin,eng|Bible+other|Automatically translated|GNU General Public License v3.0|
|| [Lingala_Song_Lyrics](https://github.com/espoirMur/songs_lyrics_webscrap)| - |fra, lin|Scrape the content of the website www.ndombolo.co, the site have almost 30 songs in lingala and their french traduction|Web scraped| |
|| [Mburisano_Covid](https://repo.sadilar.org/handle/20.500.12185/536)| - |afr, eng, nde, sot, ssw, tsn, tso, ven, xho, zul|Corpus with limited domain|Manually translated|CC BY 3.0|
|| [Menyo20K](https://github.com/uds-lsv/menyo-20k_MT) | [Link](https://aclanthology.org/2021.mtsummit-research.6/) |yor, eng|Parallel, multidomain|News articles (JW), ted talks, movie transcripts, radio transcripts, science and technology texts, and other short articles curatedfrom the web and professional translators|non-commercial use|
|| [Minangkabau corpora](https://github.com/fajri91/minangNLP) | [Link](https://aclanthology.org/2020.paclic-1.17.pdf) |min_Latn, ind|Parallel sentences|Annotated|MIT License|
|| [Makerere MT Corpus](https://zenodo.org/record/5089560#.Y00i3uxBw-S) | - |lug, eng|Parallel sentences|Annotated|CC BY 4.0|
|| [isiZulu](https://zenodo.org/record/5035171#.YaippvHMJDZ)| - |zul,eng|English sentences, sampled from News Crawl datasets that were translated into isiZulu|Annotated|CC BY 4.0|
|| [XhosaNavy](https://opus.nlpl.eu/XhosaNavy.php) | [Link](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=25ca4a36df2955b345634b5f8a6b6bb66a774b3c) |xho,eng|South African Navy parallel corpus | | |
||[Add](https://github.com/drelhaj/ArabicDialects) | [Link](http://www.lrec-conf.org/proceedings/lrec2018/pdf/237.pdf) |arz, afb, ajp, apc|Dialects, arabic commentaries|Annotated|Freely available for research purposes|
|| [AfriBERTa](https://huggingface.co/datasets/castorini/afriberta-corpus)| [Link](https://aclanthology.org/2021.mrl-1.11.pdf) |amh, hau, ibo, orm, pcm, som, swa, tir, yor|mostly BBC, some Common Crawl| |Apache License 2.0|
|| [AfroMAFT](https://zenodo.org/record/6990611#.Y0-yU-xBw-Q) | [Link](https://aclanthology.org/2022.naacl-main.223/) ; [Link](https://aclanthology.org/2021.naacl-main.41/) | <details> afr, amh ,ara, eng, fra, hau, ibo, mlg, nya, orm, pcm, kin, sna, som, sot, swa, xho, yor, zul <summary> expand </summary> </details> |Language Adaptation Corpus| |https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/|
|| [AIFORTHAI-LotusCorpus](https://github.com/korakot/corpus/releases/download/v1.0/AIFORTHAI-LotusCorpus.zip)  | - |tha|Large vOcabualry Thai continUous Speech recognition (LOTUS) corpus| |2005 Copyright by National Electronics and Computer Technology Center (NECTEC). For more information, visit http://www.nectec.or.th/rdi/lotus|
|| [AraBench](https://alt.qcri.org/resources1/mt/arabench/) | [Link](https://aclanthology.org/2020.coling-main.447.pdf) |arz, apc, afb, ary|Translations of 'travelling phrases', blogs, tv transcripts, Bible|Pool together already available Dialectal Arabic-English resources andcurate their own evaluation sets.|Apache License 2.0|
|| [AUTSHUMATO](https://autshumato.sourceforge.net/)|  |tsn, tso|South African government domain| | |
|| [BLOOM](https://huggingface.co/datasets/sil-ai/bloom-lm)| [Link](https://arxiv.org/pdf/2210.14712.pdf) |<details> aaa, abc, ada, adq, aeu, agq, ags, ahk, aia, ajz, aka, ame, amp, amu, ann, aph, awa, awb, azn, azo, bag, bam, baw, bax, bbk, bcc, bce, bec, bef, bfd, bfm, bfn, bgf, bho, bhs, bis, bjn, bjr, bkc, bkh, bkm, bkx, bob, bod, boz, bqm, bra, brb, bri, brv, bss, bud, buo, bwt, bwx, bxa, bya, bze, bzi, cak, cbr, cgc, chd, chp, cim, clo, cmo, csw, cuh, cuv, dag, ddg, ded, dig, dje, dmg, dnw, dtp, dtr, dty, dug, eee, ekm, enb, enc, ewo, fli, fon, fub, fuh, gal, gbj, gou, gsw, guc, guz, gwc, hao, hbb, hig, hil, hla, hna, hre, hro, idt, ilo, ino, isu, jgo, jmx, jra, kak, kam, kau, kbq, kbx, kby, kek, ken, khb, kik, kin, kjb, kmg, kmr, kms, kmu, kqr, krr, ksw, kvt, kwd, kwu, kwx, kxp, kyq, laj, lan, lbr, lfa, lgg, lgr, lhm, lhu, lkb, llg, lmp, lns, loh, lsi, lts, lug, luy, lwl, mai, mam, mdr, mfh, mfj, mgg, mgm, mgo, mgq, mhx, miy, mkz, mle, mlk, mlw, mmu, mne, mnf, mnw, mot, mqj, mrn, mry, msb, muv, mve, mxu, myk, myx, mzm, nas, nco, new, nge, ngn, nhx, njy, nla, nlv, nod, nsk, nsn, nso, nst, nuj, nwe, nwi, nxa, nxl, nyo, nyu, nza, odk, oji, oki, omw, ozm, pae, pag, pbt, pce, pcg, pdu, pea, pex, pis, pkb, pmf, pnz, psp, pwg, qaa, qub, quc, quf, quz, qve, qvh, qvm, qvo, qxh, rel, rnl, roo, rue, rug, saq, sat, sdk, sea, sgd, shn, sml, snk, snl, sox, sps, ssn, stk, sxb, syw, taj, tbj, tdb, tdg, tdt, teo, tet, the, thk, thl, thy, tio, tkd, tnl, tnn, tnp, tnt, tod, tom, tpi, tpl, tpu, tsb, tsn, tso, tuv, tuz, tvs, udg, unr, ven, vif, war, wbm, wbr, wms, wni, wnk, wtk, xkg, xmd, xmg, xmm, xog, xty, yas, yav, ybb, ybh, ybi, ydd, yea, yet, yin, ymp, zaw, zlm, zuh <summary> expand </summary> </details>|Web|Crawl from Internet and filtering|CC BY 4.0|
|| [CC100](https://huggingface.co/datasets/cc100)| [Link](https://aclanthology.org/2020.acl-main.747/) ; [Link](https://aclanthology.org/2020.lrec-1.494/ ) |<details> asm, ful, grn, lim, lin, lug, nso, orm, que, roh, srd, ssw, tsn, wol <summary> expand </summary> </details>|Web|Crawl from Internet| |
|| CCNet | |  | | 
|| Clarin | |  | | 
|| [CORP.NCHLT](https://repo.sadilar.org/handle/20.500.12185/7) | - | <details> nde, nso, sot, ssw, tsn, tso, ven, xho, zul <summary> expand </summary> </details> |Various|Various|Creative Commons Attribution 2.5 South Africa License|
|| [DART](http://qufaculty.qu.edu.qa/telsayed/datasets/)  | [Link](https://aclanthology.org/L18-1579.pdf) |arz, afb, acm, apc, ary |Tweets|Annotators involved also for quality control|Publicly available|
|| [Earthlings](https://publicdata.canterbury.ac.nz/Research/Geocorpus/CCGLU_v5.0/)  |[Link](https://publicdata.canterbury.ac.nz/Research/Geocorpus/Documentation/!Paper.Corpus_of_Global_Language_Use.pdf) |<details> acu, afr, amh, amu, asm, aze, bel, ben, bod, bus, cak, cbc, cbs, cbv, ceb, chv, coe, crn, csb, cym, des, div, dop, epo, eus, fao, gle, glg, guj, gum, gym, hat, hbs, hye, ido, ilo, ipi, isl, jav, kab, kal, kan, kaz, khm, kir, knv, kpr, kur, kyc, kyq, lao, lez, lus, maa, mal, mar, maz, mkd, mlg, mlp, mon, mop, mpx, mri, mya, myy, nep, opm, ori, pan, pck, pir, poh, ptu, pus, que, sab, sah, scn, sin, sja, sme, snd, som, srd, srm, sua, swa, tat, tbc, tbz, tca, tel, tgk, tgl, tpi, tuk, ubu, udm, uig, urd, uzb, wal, wln, wol, yid, yor <summary> expand </summary> </details>|Subset of CommonCrawl|Crawl from Internet and filtering|GNU-GPL v.3 License|
|| [Flores200](https://huggingface.co/datasets/facebook/flores)  |[Link](https://arxiv.org/pdf/2207.04672.pdf) |<details> ace_Arab, ace_Latn, acm_Arab, acq_Arab, aeb_Arab, afr_Latn, ajp_Arab, aka_Latn, als_Latn, amh_Ethi, apc_Arab, arb_Arab, arb_Latn, ars_Arab, ary_Arab, arz_Arab, asm_Beng, ast_Latn, awa_Deva, ayr_Latn, azb_Arab, azj_Latn, bak_Cyrl, bam_Latn, ban_Latn, bel_Cyrl, bem_Latn, ben_Beng, bho_Deva, bjn_Arab, bjn_Latn, bod_Tibt, bos_Latn, bug_Latn, bul_Cyrl, cat_Latn, ceb_Latn, ces_Latn, cjk_Latn, ckb_Arab, crh_Latn, cym_Latn, dan_Latn, deu_Latn, dik_Latn, dyu_Latn, dzo_Tibt, ell_Grek, eng_Latn, epo_Latn, est_Latn, eus_Latn, ewe_Latn, fao_Latn, fij_Latn, fin_Latn, fon_Latn, fra_Latn, fur_Latn, fuv_Latn, gaz_Latn, gla_Latn, gle_Latn, glg_Latn, grn_Latn, guj_Gujr, hat_Latn, hau_Latn, heb_Hebr, hin_Deva, hne_Deva, hrv_Latn, hun_Latn, hye_Armn, ibo_Latn, ilo_Latn, ind_Latn, isl_Latn, ita_Latn, jav_Latn, jpn_Jpan, kab_Latn, kac_Latn, kam_Latn, kan_Knda, kas_Arab, kas_Deva, kat_Geor, kaz_Cyrl, kbp_Latn, kea_Latn, khk_Cyrl, khm_Khmr, kik_Latn, kin_Latn, kir_Cyrl, kmb_Latn, kmr_Latn, knc_Arab, knc_Latn, kon_Latn, kor_Hang, lao_Laoo, lij_Latn, lim_Latn, lin_Latn, lit_Latn, lmo_Latn, ltg_Latn, ltz_Latn, lua_Latn, lug_Latn, luo_Latn, lus_Latn, lvs_Latn, mag_Deva, mai_Deva, mal_Mlym, mar_Deva, min_Arab, min_Latn, mkd_Cyrl, mlt_Latn, mni_Beng, mos_Latn, mri_Latn, mya_Mymr, nld_Latn, nno_Latn, nob_Latn, npi_Deva, nso_Latn, nus_Latn, nya_Latn, oci_Latn, ory_Orya, pag_Latn, pan_Guru, pap_Latn, pbt_Arab, pes_Arab, plt_Latn, pol_Latn, por_Latn, prs_Arab, quy_Latn, ron_Latn, run_Latn, rus_Cyrl, sag_Latn, san_Deva, sat_Olck, scn_Latn, shn_Mymr, sin_Sinh, slk_Latn, slv_Latn, smo_Latn, sna_Latn, snd_Arab, som_Latn, sot_Latn, spa_Latn, srd_Latn, srp_Cyrl, ssw_Latn, sun_Latn, swe_Latn, swh_Latn, szl_Latn, tam_Taml, taq_Latn, taq_Tfng, tat_Cyrl, tel_Telu, tgk_Cyrl, tgl_Latn, tha_Thai, tir_Ethi, tpi_Latn, tsn_Latn, tso_Latn, tuk_Latn, tum_Latn, tur_Latn, twi_Latn, tzm_Tfng, uig_Arab, ukr_Cyrl, umb_Latn, urd_Arab, uzn_Latn, vec_Latn, vie_Latn, war_Latn, wol_Latn, xho_Latn, ydd_Hebr, yor_Latn, yue_Hant, zho_Hans, zho_Hant, zsm_Latn, zul_Latn <summary> expand </summary> </details>|Misc|Human annotated|CC-BY-SA 4.0|
|| [FrenchEwe](https://zenodo.org/record/4266935#.YaTu0fHMJDY) | - |ewe, fra|Parallel sentences|Annotated|CC-BY-4.0|
|| [Glosses](http://lcl.uniroma1.it/disambiguated-glosses/) |[Link](http://lcl.uniroma1.it/disambiguated-glosses/files/A_Large-Scale_Multilingual_Disambiguation_of_Glosses.pdf) |256 languages|Disambiguated glosses|Wikipedia, Wiktionary, WordNet, OmegaWiki and Wikidata.|CC BY-NC-SA 3.0|
|| [Habibi](http://ucrel-web.lancaster.ac.uk/habibi/)  | [Link](https://eprints.lancs.ac.uk/id/eprint/142282/1/habibi.pdf) |arz, afb, acm, ary, apd, apc|Song lyrics|Collected from the Web|Freely available for research purposes|
|| [Hindialect](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4839) | [Link](https://dspace.cuni.cz/bitstream/handle/20.500.11956/175497/120426711.pdf?sequence=1) |<details> anp, awa, ben, bgc, bhb, bhd, bho, bjj, bns, bra, gbm, guj, hin, hne, kfq, kfy, mag, mar, mis, mup, noe, pan, raj, san <summary> expand </summary> </details> |script all in Devanagari|folksongs|CC BY-NC-SA 4.0|
|| IITB | |  | | 
|| [Indiccorp](https://ai4bharat.iitm.ac.in/corpora)  | [Link](https://aclanthology.org/2020.findings-emnlp.445.pdf) |asm, ben, guj, kan, mal, mar, ory, pan, tel|Web|Web crawled|CC BY-NC-SA 4.0|
|| JParaCrawl | |  | | 
| No | [JW](https://www.jw.org/en/) | - | | Religious | Web crawled | Private
|| [LeipzigData](https://wortschatz.uni-leipzig.de/en/download)  | [Link](http://www.lrec-conf.org/proceedings/lrec2012/pdf/327_Paper.pdf) |<details> aar, ace, ach, aka, als, als-al, als-sqi, anw, arg, arz, asm, ast, aym, aze, azj, azj-az, bak, bam, ban, ban-id, bar, bcl, bem, bew, bih, bik, bjn, bjn-id, bod, bos, bpy, bua, bug, cdo, ceb, che, chv, ckb, cos, csb, diq, div, div-mv, dsb, dyu, ekk, emk, eml, ewe, ext, fao, fao-fo, fon, frr, fuc, ful, gan, glk, glv, gom, grn, gsw, gsw-ch, guj, hat, hat-ht, hbs, hbs-rs, hif, hil, hsb, ibb, ibo, ido, ile, ilo, ina, kab, kal, kal-gl, kas, kbd, kde, kea, khk, kik, kin, kng, knn, knn-in, koi, kom, kon, krc, ksh, ksw, lad, lgg, lim, lim-nl, lin, lmo, ltz, ltz-lu, lug, lup, lus, lus-in, lvs, mad, mad-id, mai, mhr, min, min-id, mkw, mlt, mos, mri, mri-nz, mrj, mwl, myv, mzn, nan, nap-tara, nav, nbl, ndo, nds, nds-nl, new, ngl, nno, nno-no, nob, nob-com, nob-no, nso, nso-za, nya, nyn, oci, oci-fr, orm, oss, pag, pam, pap, pcm, pfl, plt, pms, pnb, pnt, pus, roh, roh-ch, rom, rue, rue-ua, run, sah, san, scn, sco, seh, sgs, sin, skr, sme, sme-no, smi, sna, sna-zw, snd, snk, som, sot, sot-za, srd, ssw, ssw-za, suk, sun, sun-id, sus, swa, swh, szl, tat, tel, tem, tgk, tgk-tj, tgk-uz, tgl, tir, tiv, tsn, tsn-bw, tsn-za, tso, tso-za, tuk, tuk-tm, tum, tyv, udm, uig, uzb, uzn-uz, vec, vec-br, vec-hr, ven, ven-za, vls, vol, vro, war, wln, wol, wuu, xmf, ydd, yid, yor, zea, zha, zsm, zul, zul-za <summary> expand </summary> </details>|Wikipedia, News, WebCrawl corpora of different years|Crawl from Internet|CC BY-NC-SA 3.0|
|| Lindat | |  | | 
|| [Lyrics](https://lyricstranslate.com/) | |<details> aar, abq, adq, ady, agx, aih, ain, aka, akk, ale, ami, ang, arg, arn, arp, asm, ast, aym, bak, bam, bci, bft, bfy, bgc, bhb, bho, bik, bis, bns, bod, bsk, bvd, bya, cab, cbk, cha, che, chg, cho, chr, chv, ckm, cnr, com, cor, cre, crh, csb, ctg, dak, dng, doi, dua, dum, dyu, dzo, enm, evn, ewe, ewo, ext, fao, fij, fon, frm, fro, fur, gag, gbm, gil, gla, glg, glk, gmh, goh, gon, got, gqn, grc, grt, hif, hil, hlb, hne, hop, hsb, ido, ina, inh, ist, izh, jam, jbo, kab, kas, kbd, kca, kdr, kea, kfy, kha, kik, kin, kio, kir, kjh, kmb, kok, kom, kon, krc, krl, kru, ksh, kum, lad, lbj, ldd, lij, lin, lki, lkt, lmo, ltg, lzh, lzz, mag, mah, mai, mbx, mby, min, mjw, mnc, mni, mnk, mns, moh, mos, mrg, mus, mwl, mxi, nan, nap, nav, nds, new, nio, niu, nog, non, nys, oci, odt, ohu, orm, ory, ota, pag, pap, pau, pcd, pcm, pdt, pjt, pli, pnt, pot, que, qya, raj, rar, rhg, roh, rom, rop, rtm, rup, sag, sah, sat, scn, sco, sdc, sel, sgh, sgs, sjn, skr, slr, smn, srn, ssw, sux, syl, szl, tah, tat, tbh, tcy, tet, tir, tlh, tpi, tsn, tuk, twe, twi, tyv, tzo, udm, uig, uki, ulk, unr, vec, ven, vep, vot, wbl, wol, wym, xal, xmf, xno, xxb, yux, zap, zha, zpu, zun, zza <summary> expand </summary> </details>|Lyrics| | |
|| [MaCoCu](https://macocu.eu/) | [Link](https://research.rug.nl/en/publications/macocu-massive-collection-and-curation-of-monolingual-and-bilingu) |mlt| |Crawl from Internet and filtering| |
|| [MC4](https://huggingface.co/datasets/mc4) | [Link](https://arxiv.org/pdf/1910.10683.pdf) | <details> aze, ceb, cos, fil, guj, hat, haw, hmn, ibo, ltz, mlt, mri, nya, smo, sna, sot, sun, tgk, yor, zul <summary> expand </summary> </details>|Web|Crawl from Internet|odc-by|
|| [MoT](https://github.com/bltlab/mot)| [Link](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.224.pdf) |kin, lin, nde, orm, bod, tir|Data collected from Voice of America (VOA) news websites| |MIT License|
|| [MTData](https://github.com/thammegowda/mtdata) ||||||
|| [Nart/abkhaz](https://huggingface.co/datasets/Nart/abkhaz_text) | - |abk|multiple sources| |Creative Commons Universal Public Domain License|
|| [Ndc_without_informant_codes](http://tekstlab.uio.no/nota/scandiasyn/dialect_data_collection.html) | |dan, fao, isl, ovd, swe|Nordic Dialect Corpus comprises recorded speech data from the Nordic countries, in languages that belong to the North Germanic language family.|Various http://tekstlab.uio.no/nota/scandiasyn/dialect_data_collection.html |CC BY-NC-SA 4.0|
|| [NLLB_seed](https://github.com/facebookresearch/flores/blob/main/nllb_seed/README.md)  | [Link](https://arxiv.org/abs/2207.04672) |<details> ace_Arab, ace_Latn, ary, arz, bam, ban, bho, bja_Arab, bjn_Latn, bug, crh, dik, dzo, fur, fuv, grn, hne, kas_Latn, kas_Deva, knc_Arab, knc_Latn, lij, lim, lmo, ltg, mag, mni, mri, nus, prs, pbt, scn, shn, srd, szl, taq_Tfng, taq_Latn, tzm, vec <summary> expand </summary> </details>|Collection of topics in different fields of knowledge and human activity|Professionally-translated sentences in the Wikipedia domain|CC-BY-SA 4.0|
|| [OfisPublik](http://opus.nlpl.eu/OfisPublik-v1.php) | [Link](https://aclanthology.org/2009.eamt-1.29.pdf) ; [Link](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf)|bre|Texts from the Ofis Publik ar Brezhoneg (Breton Language Board) provided by Francis Tyers| | |
|| [OPUS](https://opus.nlpl.eu/) | [Link](https://aclanthology.org/L12-1246/) | |  | | 
|| [OSCAR](https://oscar-project.org/) | [Link](https://ids-pub.bsz-bw.de/frontdoor/deliver/index/docId/9021/file/Suarez_Sagot_Romary_Asynchronous_Pipeline_for_Processing_Huge_Corpora_2019.pdf) |<details> als, arg, arz, asm, ast, ava, aze, bak, bho, bod, bos, bpy, bxr, ceb, che, chv, ckb, cor, diq, div, dsb, eml, gom, grn, guj, hbs, hsb, ido, ilo, ina, jbo, kom, krc, lez, lim, lmo, ltz, mai, mhr, min, mlt, mrj, mzn, nah, nds, new, nno, oci, oss, pms, pnb, que, sah, scn, sun, tat, tgk, tuk, vol, war, wln, wuu, xal, xmf, yor <summary> expand </summary> </details>|2201 version: https://huggingface.co/datasets/oscar-corpus/OSCAR-2201|Crawl from Internet and filtering|CC BY 4.0|
|| ParaCrawl | |  | | 
|| [Parallel Bible Corpus]() | [Link](http://cysouw.de/home/presentations_files/cysouwmayerPARALLELLREC.pdf) | | Religious | |You can contact Michael Cysouw, Philipps University of Marburg, to request access to the Parallel Bible Corpus for academic purposes. | 
|| [Parallel Corpora for Ethiopian Languages](https://github.com/AAUThematic4LT/Parallel-Corpora-for-Ethiopian-Languages) | [Link](https://aclanthology.org/W18-3812.pdf)|  amh, orm, tir|Parallel sentences, religious domain|Automatically curated|CC-BY-4.0|
|| Phontron | |  | | 
|| [QADI](https://alt.qcri.org/resources/qadi) |[Link](https://arxiv.org/abs/2005.06557) | <details> afb, abv, arq, arz, acm, apc, ary, acx, ajp, apd, aeb <summary> expand </summary> </details> |Tweets|Tweets|Apache License 2.0|
|| [Quechua-IIC](https://huggingface.co/datasets/Llamacha/monolingual-quechua-iic)  |[Link](https://aclanthology.org/2022.deeplo-1.1.pdf) |que|multiple sources| |Apache License 2.0|
|| [Shami](https://github.com/GU-CLASP/shami-corpus/tree/master/Data) | [Link](https://aclanthology.org/L18-1576.pdf) |apc, ajp|Several topics from regular conversations such as politics, education, society, health care, house keeping and others|Automatic and manual approaches|Apache License 2.0|
|| [SLI_GalWeb.1.0](https://ilg.usc.gal/download/SLI_Galician_Corpora/SLI_GalWeb.1.0.tar.gz) | [Link](https://aclanthology.org/L18-1367.pdf) |glg|Galician political party, newspaper, government official website|Crawling data from many Web data sources|CC BY 4.0|
|| Stanford NLP | |  | | 
|| [StatMT](https://statmt.org/) | | |  | | 
|| [Tatoeba](https://tatoeba.org/en/) |  |<details> abk, acm, ady, afb, afh, afr, aii, ain, ajp, akl, aln, alt, amh, ang, aoz, apc, ara, arg, arq, ary, arz, asm, ast, avk, awa, ayl, aym, aze, bak, bal, bam, ban, bar, bcl, bel, ben, ber, bfz, bho, bis, bjn, bod, bom, bos, bre, brx, bua, bul, bvy, bzt, cat, cay, cbk, ceb, ces, cha, che, chg, chn, cho, chr, chv, cjy, ckb, ckt, cmn, cmo, cor, cos, cpi, crh, crk, crs, csb, cycl, cym, cyo, dan, deu, diq, div, dng, drt, dsb, dtp, dws, egl, ell, emx, eng, enm, epo, est, eus, evn, ewe, ext, fao, fij, fin, fkv, fra, frm, fro, frr, fry, fuc, fur, fuv, gaa, gag, gan, gbm, gcf, gil, gla, gle, glg, glv, gom, gos, got, grc, grn, gsw, guc, guj, hak, hat, hau, haw, hax, hbo, hdn, heb, hif, hil, hin, hnj, hoc, hrv, hrx, hsb, hsn, hun, hye, iba, ibo, ido, igs, iii, ike, ile, ilo, ina, ind, isl, ita, izh, jam, jav, jbo, jdt, jpa, jpn, kaa, kab, kal, kam, kan, kas, kat, kaz, kek, kha, khm, kin, kir, kiu, kjh, klj, kmr, knc, koi, kor, kpv, krc, krl, ksh, kum, kxi, kzj, laa, lad, lao, lat, ldn, lfn, lij, lim, lin, lit, liv, lkt, lld, lmo, lou, ltg, ltz, lug, lut, lvs, lzh, lzz, mad, mah, mai, mal, mar, max, mdf, mfa, mfe, mgm, mhr, mic, mik, min, mkd, mlg, mlt, mnc, mni, mnr, mnw, moh, mon, mri, mrj, mus, mvv, mwl, mww, mya, myv, nah, nan, nau, nav, nch, nds, new, ngt, ngu, niu, nld, nlv, nnb, nno, nob, nog, non, nov, npi, nst, nus, nya, nys, oar, oci, ofs, oji, ood, ori, orv, osp, oss, osx, ota, otk, pag, pal, pam, pan, pap, pau, pcd, pdc, pes, pfl, phn, pli, pms, pnb, pol, por, ppl, prg, pus, quc, que, qxq, qya, rap, rel, rhg, rif, roh, rom, ron, rue, run, rus, ryu, sag, sah, san, sat, scn, sco, sdh, sgs, shi, shs, shy, sin, sjn, skr, slk, slv, sma, sme, smo, sna, snd, som, sot, spa, sqi, srd, srn, srp, ssw, stq, sun, sux, swc, swe, swg, swh, syc, szl, tah, tam, tat, tel, tet, tgk, tgl, tha, thv, tig, tir, tkl, tlh, tly, tmr, tmw, toi, tok, ton, tpi, tpw, tsn, tso, tts, tuk, tur, tvl, tyv, tzl, udm, uig, ukr, umb, urd, urh, uzb, vec, vep, vie, vol, vro, war, wln, wol, wuu, xal, xho, xmf, xqa, yid, yor, yua, yue, zea, zgh, zlm, zsm, zul, zza <summary> expand </summary> </details>|180922 version|Voluntary contributions of thousands of members|under various Creative Commons licenses: https://tatoeba.org/en/terms_of_use#section-6|
|| [TeDDi](https://github.com/MorphDiv/TeDDi_sample) | [Link](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.123.pdf) |<details> abk, aey, amp, ape, apu, arn, arz, ayz, bmi, bsk, bsn, cha, ckt, crk, dgz, dni, fij, gni, gry, gug, gyd, hae, hau, hix, hnj, imn, jac, kal, kan, kew, kgo, khk, kio, kjq, kut, laj, lue, lvk, mig, mph, mya, myh, myp, mzh, naq, ote, pav, plt, pwn, qvi, ram, rap, rma, sag, spp, swh, tiw, tml, tzm, vma, wba, wic, wyb, xsu, yad, yaq, yor, zoc, zul <summary> expand </summary> </details>|Collection of different sources (see paper)|Language identification and filtering|CC BY-NC-SA 4.0|
|| [TICO](https://tico-19.github.io/) | [Link](https://tico-19.github.io/data/paper/ticopaper.pdf) |<details> amh, ara, ben, ckb, din, eng, fas, fra, fuv, hau, hin, ind, khm, knc, kmr, lug, lin, mar, msa, mya, npi, nus, orm, prs, por, pus, rus, kinn, som, spa, swh, tam, tir_et, tir_er, tgl, urd, zho, zul <summary> expand </summary> </details>|COVID-19 materials for a variety of the world’s languages to be used by professional translators and for training SOTA Machine Translation (MT) models| Annotated|CC0 1.0 Universal|
|| [TIL](https://github.com/turkic-interlingua/til-mt) | [Link](https://aclanthology.org/2021.emnlp-main.475/) | <details> aze, bak, chv, eng, kaz, kir, rus, tuk, tur, tat, uig, uzb <summary> expand </summary> </details> | Large-scale parallel corpus combinin gmost of the public datasets for 22 Turkic languages | Automatically collected|CC BY-NC-SA 4.0|
|| Tilde | |  | | 
|| [w2c](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0022-6133-9)  | - |122 languages|Corpus|Automatically collected from wikipedia and the web|CC BY-SA 3.0|
|| WAT | |  | | 
|| [Wikipedia](https://huggingface.co/datasets/wikipedia)| - |<details> aar, abk, ace, ady, aka, als, ang, arc, arg, arz, asm, ast, atj, ava, aym, aze, bak, bam, bar, bcl, ben, bih, bis, bjn, bod, bos, bpy, bre, bug, bul, bxr, cbk, cdo, ceb, cha, che, cho, chr, chu, chv, chy, ckb, cor, cos, cre, crh, csb, din, diq, div, dsb, dty, dzo, eml, ewe, ext, fao, fij, frp, frr, ful, fur, gag, gan, glg, glk, glv, gom, gor, got, grn, guj, hak, hat, haw, hbs, hif, hmo, hsb, ibo, ido, iii, iku, ile, ilo, ina, inh, ipk, isl, jam, jbo, jpn, kaa, kab, kal, kas, kbd, kbp, kik, kin, koi, kom, kon, krc, ksh, kua, lad, lbe, lez, lfn, lij, lim, lin, lmo, lrc, ltg, ltz, lug, lzh, mah, mai, mdf, mhr, min, mlt, mri, mrj, mus, mwl, myv, mzn, nah, nan, nap, nau, nav, ndo, nds, new, nno, nov, nrm, nso, nya, oci, olo, orm, oss, pag, pam, pan, pap, pcd, pdc, pfl, pih, pli, pms, pnb, pnt, que, rmy, roh, rue, run, rup, rus, sag, sah, sat, scn, sco, sgs, sme, smo, sna, sot, srd, srn, ssw, stq, sun, szl, tah, tat, tcy, tet, tgk, tir, ton, tpi, tsn, tso, tuk, tum, twi, tyv, udm, vec, ven, vep, vls, vol, vro, war, wln, wol, wuu, xal, xmf, yor, yue, zea, zha, zul <summary> expand </summary> </details>|20221001|Wikipedia|CC BY-NC-SA 3.0|
|| WikiMatrix | |  | | 
|| [Workshop on NER for South and South East Asian Languages](https://ltrc.iiit.ac.in/ner-ssea-08/index.cgi?topic=5) | [Link](https://aclanthology.org/I08-5003.pdf) |ben, ori, urd| | |Data can be freely used for non-profit research work under the Creative Commons License.|
|| [XLSum](https://huggingface.co/datasets/csebuetnlp/xlsum)  |[Link](https://arxiv.org/pdf/2106.13822.pdf) |aze, guj, ibo, orm, run, tir, yor|BBC| |CC BY-NC-SA 4.0|
| | | | | | | |


## Training and Evalutaion Code

**Notice**: The missing part of the code will be completed shortly.

### Prerequisites

We use two settings due to package conflict:

- Major: Python 3.9, `requirements.txt`
- Sentence retrieval: Python 3.6, `evaluation/retrieval/requirements.txt`

### Data preparation

For training both tokenizer and model of Glot500-m, we need to prepare a **balanced** corpus covering all languages.

Go to 'preprocessing/' and run:

```
bash merge_files.sh
```

We set `--scale 1` for tokenizer, `--scale 30` for model.  

### Vocabulary Extension

Go to 'tokenization/' and run:

```
bash train.sh
```

### Continued Pretraining

Go to 'modeling/' and run:

```
bash train_bash.sh
```

##  Glot500-m Evaluation
We provide in-depth evaluation of Glot500-m model and baselines in the paper. Each number is an average over head languages, tail languages and all languages. See the paper for detailed results per task and language. Glot500-m outperforms XLM-R-B (base) in all tasks for head (except for POS) and tail languages and XLM-R-L (large) for tail languages. Best result per task/language set is in bold.

||   tail    | tail | tail | head | head | head | all | all | all |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | XLM-R-B | XLM-R-L | [Glot500-m](https://huggingface.co/cis-lmu/glot500-base) | XLM-R-B | XLM-R-L | [Glot500-m](https://huggingface.co/cis-lmu/glot500-base) | XLM-R-B | XLM-R-L | [Glot500-m](https://huggingface.co/cis-lmu/glot500-base) |
| Pseudoperplexity | 304.2 | 168.6 | **12.2** | 12.5 | **8.4** | 11.8 | 247.8 | 136.4 | **11.64** |
| Sentence Retrieval Tatoeba (Top 10 Acc.) | 32.6 | 33.6 | **59.8** | 66.2 | 71.1 | **75.0** | 56.6 | 60.4 | **70.7** |
| Sentence Retrieval Bible (Top 10 Acc.) | 7.4 | 7.1 | **43.2** | 54.2 | 58.3 | **59.0** | 19.3 | 20.1 | **47.3** |
| Text Classification (F1) | 13.7 | 13.9 | **46.6** | 51.3 | **60.5** | 54.7 | 23.3 | 25.8 | **48.7** |
| NER (F1) | 47.5 | 51.8 | **60.7** | 61.8 | **66.0** | 63.9 | 55.3 | 59.5 | **62.4** |
| POS (F1) | 41.7 | 43.5 | **62.3** | 76.4 | **78.4** | 76.0 | 65.8 | 67.7 | **71.8** |
| Roundtrip Alignment (Acc.) | 2.57 | 3.13 | **4.45** | 3.42 | 4.06 | **5.46** | 2.77 | 3.34 | **4.68** |

## Citation

 If you find our model, data or the overview of data useful for your research, please cite:

```
@inproceedings{imani-etal-2023-glot500,
    title = "Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages",
    author = " ImaniGooghari, Ayyoob and Lin, Peiqin and Kargaran, Amir Hossein and Severini, Silvia and Sabet, Masoud Jalili and Kassner, Nora and Ma, Chunlan and Schmid, Helmut and Martins, André and Yvon, François and  Sch{\"u}tze, Hinrich",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics.",
    year = "2023",
}
```

## Acknowledgements

This repository is built on top of [transformers](https://github.com/huggingface/transformers) and [xtreme](https://github.com/google-research/xtreme).

