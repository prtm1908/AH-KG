# AH-KG

## OpenIE Commands

```bash
cd /Users/pratham/.stanfordnlp_resources/stanford-corenlp-4.5.3
```

```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,depparse,natlog,coref,openie" -port 9000 -timeout 30000
```