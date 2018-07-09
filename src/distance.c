//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define MAX_STRING 1100

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const char split_pattern = '\t';
long long vocab_hash_size = 30000000;
int has_title = 0;
int max_sense_num = 10;
int max_label_size = 3653051;
int layer_size = 0;

typedef float real;                    // Precision of float numbers

struct vocab_item {
    long long cn;
    char *item;
};

struct vocab_title {
    //sense_num for NIL, entity_num for KB
    int sense_num, entity_num, max_entity_index_num;
    int *entity_index;
    char *item;
};

struct model_var {
    struct vocab_item *vocab;
    int *vocab_hash, vocab_hash_size;
    long long vocab_size, vocab_max_size;
    real *vec;
    char name[MAX_STRING], read_file[MAX_STRING];
}words, entities;

struct model_var2 {
    char name[MAX_STRING], read_file[MAX_STRING];
    //for titles
    struct vocab_title *vocab;
    int *vocab_hash, *cluster_size, *et_cluster_size;
    int vocab_hash_size;
    long long vocab_max_size, vocab_size;
    real *vec;
    real *mu;
    //for the sense linking to KB
    real *et_vec;
    real *et_mu;
}titles;

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % words.vocab_hash_size;
    return hash;
}

// Returns hash value of a title
int GetTitleHash(char *title) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(title); a++) hash = hash * 257 + title[a];
    hash = hash % titles.vocab_hash_size;
    return hash;
}

// Returns hash value of an entity
int GetEntityHash(char *entity) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(entity); a++) hash = hash * 257 + entity[a];
    hash = hash % entities.vocab_hash_size;
    return hash;
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    words.vocab[words.vocab_size].item = (char *)calloc(length, sizeof(char));
    strcpy(words.vocab[words.vocab_size].item, word);
    words.vocab[words.vocab_size].cn = 0;
    words.vocab_size++;
    // Reallocate memory if needed
    if (words.vocab_size + 2 >= words.vocab_max_size) {
        words.vocab_max_size += 100000;
        words.vocab = (struct vocab_item *)realloc(words.vocab, words.vocab_max_size * sizeof(struct vocab_item));
    }
    hash = GetWordHash(word);
    while (words.vocab_hash[hash] != -1) hash = (hash + 1) % words.vocab_hash_size;
    words.vocab_hash[hash] = words.vocab_size - 1;
    return words.vocab_size - 1;
}

void AddEntityIndexToTitle(int title_index, int entity_index){
    titles.vocab[title_index].entity_index[titles.vocab[title_index].entity_num] = entity_index;
    titles.vocab[title_index].entity_num ++;
    // Reallocate memory if needed
    if (titles.vocab[title_index].entity_num + 2 >= titles.vocab[title_index].max_entity_index_num) {
        titles.vocab[title_index].max_entity_index_num += 10;
        titles.vocab[title_index].entity_index = (int *)realloc(titles.vocab[title_index].entity_index, titles.vocab[title_index].max_entity_index_num * sizeof(int));
    }
}

// Adds a title to the vocabulary
int AddTitleToVocab(char *title) {
    //add item string
    unsigned int hash, length = strlen(title) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    titles.vocab[titles.vocab_size].item = (char *)calloc(length, sizeof(char));
    strcpy(titles.vocab[titles.vocab_size].item, title);
    //reset entitynum, sensenum, max entity index num
    titles.vocab[titles.vocab_size].entity_num = 0;
    titles.vocab[titles.vocab_size].sense_num = 0;
    titles.vocab[titles.vocab_size].max_entity_index_num = 10;
    //allocate memory for entity index
    titles.vocab[titles.vocab_size].entity_index = (int *)calloc(titles.vocab[titles.vocab_size].max_entity_index_num, sizeof(int));
    
    titles.vocab_size++;
    // Reallocate memory if needed
    if (titles.vocab_size + 2 >= titles.vocab_max_size) {
        titles.vocab_max_size += 100000;
        titles.vocab = (struct vocab_title *)realloc(titles.vocab, titles.vocab_max_size * sizeof(struct vocab_title));
    }
    hash = GetTitleHash(title);
    while (titles.vocab_hash[hash] != -1) hash = (hash + 1) % titles.vocab_hash_size;
    titles.vocab_hash[hash] = titles.vocab_size - 1;
    
    return titles.vocab_size - 1;
}

// Adds an entity to the vocabulary
int AddEntityToVocab(char *entity) {
    unsigned int hash, length = strlen(entity) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    entities.vocab[entities.vocab_size].item = (char *)calloc(length, sizeof(char));
    strcpy(entities.vocab[entities.vocab_size].item, entity);
    entities.vocab[entities.vocab_size].cn = 0;
    entities.vocab_size++;
    // Reallocate memory if needed
    if (entities.vocab_size + 2 >= entities.vocab_max_size) {
        entities.vocab_max_size += 100000;
        entities.vocab = (struct vocab_item *)realloc(entities.vocab, entities.vocab_max_size * sizeof(struct vocab_item));
    }
    hash = GetEntityHash(entity);
    while (entities.vocab_hash[hash] != -1) hash = (hash + 1) % entities.vocab_hash_size;
    entities.vocab_hash[hash] = entities.vocab_size - 1;
    return entities.vocab_size - 1;
}


// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchTextVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (words.vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, words.vocab[words.vocab_hash[hash]].item)) return words.vocab_hash[hash];
        hash = (hash + 1) % words.vocab_hash_size;
    }
    return -1;
}

// Returns position of a title in the joint model vocab; if the title is not found, returns -1
int SearchTitleVocab(char *title) {
    unsigned int hash = GetTitleHash(title);
    while (1) {
        if (titles.vocab_hash[hash] == -1) return -1;
        if (!strcmp(title, titles.vocab[titles.vocab_hash[hash]].item)) return titles.vocab_hash[hash];
        hash = (hash + 1) % titles.vocab_hash_size;
    }
    return -1;
}

// Returns position of an entity in the vocabulary; if the entity is not found, returns -1
int SearchKgVocab(char *entity) {
    unsigned int hash = GetEntityHash(entity);
    while (1) {
        if (entities.vocab_hash[hash] == -1) return -1;
        if (!strcmp(entity, entities.vocab[entities.vocab_hash[hash]].item)) return entities.vocab_hash[hash];
        hash = (hash + 1) % entities.vocab_hash_size;
    }
    return -1;
}

void LoadWordVector(){
    FILE *f;
    long long a, b, c, word_index;
    char tmp_word[MAX_STRING];
    real *tmp_syn0, len=0;
    long long vocab_size = 0;
    
    //init words hash
    words.vocab_hash_size = 30000000;
    words.vocab_hash = (int *)calloc(words.vocab_hash_size, sizeof(int));
    for (a = 0; a < words.vocab_hash_size; a++) words.vocab_hash[a] = -1;
    
    f = fopen(words.read_file, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return;
    }
    fscanf(f, "%lld", &vocab_size);
    fscanf(f, "%d", &layer_size);
    
    //init words vocab
    words.vocab = (struct vocab_item *)calloc(vocab_size, sizeof(struct vocab_item));
    words.vocab_size = 0;
    words.vocab_max_size = vocab_size;
    
    //init words vec
    a = posix_memalign((void **)&(words.vec), 128, (long long)vocab_size * layer_size * sizeof(real));
    if (words.vec == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    tmp_syn0 = (real *)calloc(layer_size, sizeof(real));
    for (b = 0; b < vocab_size; b++) {
        a = 0;
        while (1) {
            tmp_word[a] = fgetc(f);
            if (feof(f) || (tmp_word[a] == '\t')) break;
            if ((a < MAX_STRING) && (tmp_word[a] != '\n')) a++;
        }
        tmp_word[a] = 0;
        word_index = AddWordToVocab(tmp_word);
        if (word_index!=-1){
            for (c = 0; c < layer_size; c++) fread(&words.vec[c + word_index * layer_size], sizeof(float), 1, f);
            //norm
            len = 0;
            for (c = 0; c < layer_size; c++) len += words.vec[c + word_index * layer_size] * words.vec[c + word_index * layer_size];
            len = sqrt(len);
            for (c = 0; c < layer_size; c++) words.vec[c + word_index * layer_size] /= len;
        }
        else
            for (a = 0; a < layer_size; a++) fread(tmp_syn0, sizeof(float), 1, f);
    }
    fclose(f);
}

void LoadEntityVector(){
    FILE *f;
    long long a, b, c, entity_index, vocab_size;
    char tmp_entity[MAX_STRING];
    real *tmp_syn0, len = 0;
    
    //init words hash
    entities.vocab_hash_size = 30000000;
    entities.vocab_hash = (int *)calloc(entities.vocab_hash_size, sizeof(int));
    for (a = 0; a < entities.vocab_hash_size; a++) entities.vocab_hash[a] = -1;

    f = fopen(entities.read_file, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return;
    }
    fscanf(f, "%lld", &vocab_size);
    fscanf(f, "%d", &layer_size);
    
    //init words vocab
    entities.vocab = (struct vocab_item *)calloc(vocab_size, sizeof(struct vocab_item));
    entities.vocab_size = 0;
    entities.vocab_max_size = vocab_size;
    
    //init entity vec
    a = posix_memalign((void **)&(entities.vec), 128, (long long)vocab_size * layer_size * sizeof(real));
    if (entities.vec == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    tmp_syn0 = (real *)calloc(layer_size, sizeof(real));
    for (b = 0; b < vocab_size; b++) {
        a = 0;
        while (1) {
            tmp_entity[a] = fgetc(f);
            if (feof(f) || (tmp_entity[a] == '\t')) break;
            if ((a < MAX_STRING) && (tmp_entity[a] != '\n')) a++;
        }
        tmp_entity[a] = 0;
        entity_index = AddEntityToVocab(tmp_entity);
        if (entity_index!=-1){
            for (c = 0; c < layer_size; c++) fread(&entities.vec[c + entity_index * layer_size], sizeof(float), 1, f);
            //norm
            len = 0;
            for (c = 0; c < layer_size; c++) len += entities.vec[c + entity_index * layer_size] * entities.vec[c + entity_index * layer_size];
            len = sqrt(len);
            for (c = 0; c < layer_size; c++) entities.vec[c + entity_index * layer_size] /= len;

        }
        else
            for (c = 0; c < layer_size; c++) fread(tmp_syn0, sizeof(float), 1, f);
    }
    fclose(f);
}

void LoadTitleVector(){
    FILE *f;
    long long a, b, c, e, title_index, entity_index, vocab_size;
    int sense_num = 0, entity_num = 0;
    char tmp_title[MAX_STRING], tmp_entity[MAX_STRING];
    real *tmp_syn0, len = 0;
    char d;
    
    //init titile hash
    titles.vocab_hash_size = 30000000;
    titles.vocab_hash = (int *)calloc(titles.vocab_hash_size, sizeof(int));
    for (a = 0; a < titles.vocab_hash_size; a++) titles.vocab_hash[a] = -1;
    
    f = fopen(titles.read_file, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return;
    }
    fscanf(f, "%lld", &vocab_size);
    fscanf(f, "%d", &layer_size);
    fscanf(f, "%d", &max_sense_num);
    
    //init titile vocab
    titles.vocab = (struct vocab_title *)calloc(vocab_size, sizeof(struct vocab_title));
    titles.vocab_size = 0;
    titles.vocab_max_size = vocab_size;
    
    //title entity embedding for KB
    a = posix_memalign((void **)&(titles.et_vec), 128, (long long)entities.vocab_size * layer_size * sizeof(real));
    if (titles.et_vec == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    //title entity cluster
    a = posix_memalign((void **)&(titles.et_mu), 128, (long long)entities.vocab_size * layer_size * sizeof(real));
    if (titles.et_mu == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    //title entity cluster size
    titles.et_cluster_size = (int *)calloc(entities.vocab_size, sizeof(int));
    
    //title sense embedding for NIL
    a = posix_memalign((void **)&(titles.vec), 128, (long long)vocab_size * max_sense_num * layer_size * sizeof(real));
    if (titles.vec == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    //sense cluster center
    a = posix_memalign((void **)&(titles.mu), 128, (long long)vocab_size * max_sense_num * layer_size * sizeof(real));
    if (titles.mu == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    //sense cluster size
    titles.cluster_size = (int *)calloc(vocab_size * max_sense_num, sizeof(int));
    
    tmp_syn0 = (real *)calloc(layer_size, sizeof(real));
    for (b = 0; b < vocab_size; b++) {
        a = 0;
        while (1) {
            tmp_title[a] = fgetc(f);
            if (feof(f) || (tmp_title[a] == '\t')) break;
            if ((a < MAX_STRING) && (tmp_title[a] != '\n')) a++;
        }
        tmp_title[a] = 0;
        
        title_index = AddTitleToVocab(tmp_title);
        fscanf(f, "%d%c%d%c", &sense_num, &d, &entity_num, &d);
        for (e = 0; e < sense_num; e++){
            titles.vocab[title_index].sense_num ++;
            for (c = 0; c < layer_size; c++) fread(&titles.vec[c + e * layer_size + title_index * max_sense_num * layer_size], sizeof(float), 1, f);
            //norm
            len = 0;
            for (c = 0; c < layer_size; c++) len += titles.vec[c + e * layer_size + title_index * max_sense_num * layer_size] * titles.vec[c + e * layer_size + title_index * max_sense_num * layer_size];
            len = sqrt(len);
            for (c = 0; c < layer_size; c++) titles.vec[c + e * layer_size + title_index * max_sense_num * layer_size] /= len;
            
            for (c = 0; c < layer_size; c++) fread(&titles.mu[c + e * layer_size + title_index * max_sense_num * layer_size], sizeof(float), 1, f);
            //norm
            len = 0;
            for (c = 0; c < layer_size; c++) len += titles.mu[c + e * layer_size + title_index * max_sense_num * layer_size] * titles.mu[c + e * layer_size + title_index * max_sense_num * layer_size];
            len = sqrt(len);
            for (c = 0; c < layer_size; c++) titles.mu[c + e * layer_size + title_index * max_sense_num * layer_size] /= len;
        }
        fscanf(f, "%c", &d);
        // read title entity embedding
        for (e = 0; e < entity_num; e++){
            a = 0;
            while (1) {
                tmp_entity[a] = fgetc(f);
                if (feof(f) || (tmp_entity[a] == '\t')) break;
                if ((a < MAX_STRING) && (tmp_entity[a] != '\n')) a++;
            }
            tmp_entity[a] = 0;
            entity_index = SearchKgVocab(tmp_entity);
            if (entity_index != -1){
                AddEntityIndexToTitle(title_index, entity_index);
                for (c = 0; c < layer_size; c++) fread(&titles.et_vec[c + entity_index * layer_size], sizeof(float), 1, f);
                //norm
                len = 0;
                for (c = 0; c < layer_size; c++) len += titles.et_vec[c + entity_index * layer_size] * titles.et_vec[c + entity_index * layer_size];
                len = sqrt(len);
                for (c = 0; c < layer_size; c++) titles.et_vec[c + entity_index * layer_size] /= len;
                
                for (c = 0; c < layer_size; c++) fread(&titles.et_mu[c + entity_index * layer_size], sizeof(float), 1, f);
                //norm
                len = 0;
                for (c = 0; c < layer_size; c++) len += titles.et_mu[c + entity_index * layer_size] * titles.et_mu[c + entity_index * layer_size];
                len = sqrt(len);
                for (c = 0; c < layer_size; c++) titles.et_mu[c + entity_index * layer_size] /= len;

                titles.et_cluster_size[entity_index] = 1;
            }
            else{
                for (c = 0; c < layer_size; c++) fread(tmp_syn0, sizeof(float), 1, f);
                for (c = 0; c < layer_size; c++) fread(tmp_syn0, sizeof(float), 1, f);
            }
            fscanf(f, "%c", &d);
        }
        
    }
    fclose(f);
}

void FindNearest(int top_n, float *vec){
    char *bestw[top_n];
    float dist, bestd[top_n];
    long long a, c, d;
    for (a = 0; a < top_n; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    for (a = 0; a < top_n; a++) bestd[a] = 0;
    for (a = 0; a < top_n; a++) bestw[a][0] = 0;
    
    //normalization
    //    len = 0;
    //    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    //    len = sqrt(len);
    //    for (a = 0; a < size; a++) vec[a] /= len;
    //compute nearest words and entities
    
    printf("\n                                              word      Cosine distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < top_n; a++) bestd[a] = -1;
    for (a = 0; a < top_n; a++) bestw[a][0] = 0;
    //compute distance with each word
    for (c = 0; c < words.vocab_size; c++) {
        a = 0;
        //skip self
        //if (index == c) continue;
        dist = 0;
        for (a = 0; a < layer_size; a++) dist += vec[a] * words.vec[a + c * layer_size];
        for (a = 0; a < top_n; a++) {
            if (dist > bestd[a]) {
                for (d = top_n - 1; d > a; d--) {
                    bestd[d] = bestd[d - 1];
                    strcpy(bestw[d], bestw[d - 1]);
                }
                bestd[a] = dist;
                strcpy(bestw[a], words.vocab[c].item);
                break;
            }
        }
    }
    for (a = 0; a < top_n; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
    
    
    printf("\n                                              entity       Cosine distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < top_n; a++) bestd[a] = -1;
    for (a = 0; a < top_n; a++) bestw[a][0] = 0;
    //compute distance with each word
    for (c = 0; c < entities.vocab_size; c++) {
        //skip those without initialized
        if (titles.et_cluster_size[c] == 0)
            continue;
        a = 0;
        //skip self
        //if (index == c) continue;
        dist = 0;
        for (a = 0; a < layer_size; a++) dist += vec[a] * titles.et_vec[a + c * layer_size];
        for (a = 0; a < top_n; a++) {
            if (dist > bestd[a]) {
                for (d = top_n - 1; d > a; d--) {
                    bestd[d] = bestd[d - 1];
                    strcpy(bestw[d], bestw[d - 1]);
                }
                bestd[a] = dist;
                strcpy(bestw[a], entities.vocab[c].item);
                break;
            }
        }
    }
    for (a = 0; a < top_n; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

void GetItem(char *item){
    long long a=0;
    printf("Enter word or entity (EXIT to break): ");
    a = 0;
    while (1) {
        item[a] = fgetc(stdin);
        if ((item[a] == '\n') || (a >= max_size - 1)) {
            item[a] = 0;
            break;
        }
        a++;
    }
}

int main(int argc, char **argv) {
    char item[max_size];
    int i, c, word_index = -1, title_index = -1, entity_index = -1,sense_index=-1;
    long long a,b;
    float word_vec[max_size], title_vec[max_size];
    int has_word = 0, has_entity = 0;
    if (argc < 2) {
        printf("\t-word_file <file>\n");
        printf("\t\tUse <file> to read the resulting word vectors\n");
        printf("\t-entity_file <file>\n");
        printf("\t\tUse <file> to read the resulting entity vectors\n");
        printf("\t-title_file <file>\n");
        printf("\t\tUse <file> to read the resulting mention vectors\n");
        printf("\nExamples:\n");
        printf("./distance -word_file ./vec_word -entity_file ./vec_entity -title_file ./vec_mention\n\n");
        return 0;
    }
    words.read_file[0] = 0;
    entities.read_file[0] = 0;
    titles.read_file[0] = 0;
    words.vocab_size = 0;
    entities.vocab_size = 0;
    titles.vocab_size = 0;
    
    if ((i = ArgPos((char *)"-word_file", argc, argv)) > 0) strcpy(words.read_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-entity_file", argc, argv)) > 0) strcpy(entities.read_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-title_file", argc, argv)) > 0) strcpy(titles.read_file, argv[i + 1]);
    
    if(0!=words.read_file[0]){
        printf("loading word vectors...\n");
        LoadWordVector();
        if(words.vocab_size>0){
            printf("Successfully load %lld words with %lld dimentions from %s\n", words.vocab_size, layer_size, words.read_file);
            has_word = 1;
        }
    }
    if(0!=entities.read_file[0]){
        printf("loading entity vectors...\n");
        LoadEntityVector();
        if(entities.vocab_size>0)
            printf("Successfully load %lld entities with %lld dimentions from %s\n", entities.vocab_size, layer_size, entities.read_file);
        has_entity = 1;
        
    }
    
    if(0!=titles.read_file[0]){
        printf("loading titile vectors...\n");
        LoadTitleVector();
        if(titles.vocab_size>0){
            printf("Successfully load %lld mentions with %lld dimentions from %s\n", titles.vocab_size, layer_size, titles.read_file);
            has_title = 1;
        }
    }
    
    if(!has_word&&!has_entity){printf("error! no words and entities loaded!");exit(1);}
    
    while(1){
        GetItem(item);
        if (!strcmp(item, "EXIT")) break;
        
        if(has_word)
            word_index = SearchTextVocab(item);
        if(has_title)
            title_index = SearchTitleVocab(item);
        
        if(word_index==-1 && title_index==-1){
            printf("Out of dictionary word or entity: %s!\n", item);
            continue;
        }
        if(word_index!=-1){
            for (a = 0; a < layer_size; a++) word_vec[a] = 0;
            for (a = 0; a < layer_size; a++) word_vec[a] += words.vec[a + word_index * layer_size];
            printf("\nWord: %s  Position in vocabulary: %d\n", item, word_index);
            FindNearest(N, word_vec);
        }
        if(title_index!=-1){
            // choose which entity
            if(titles.vocab[title_index].entity_num>1){
                printf("Entity candidates :\n");
                for(a=0;a<titles.vocab[title_index].entity_num;a++){
                    entity_index =titles.vocab[title_index].entity_index[a];
                    if(entity_index==-1)
                        continue;
                    printf("%lld : %s\n", a, entities.vocab[entity_index].item);
                }
                
                printf("please input the entity number:");
                scanf("%d",&c);
                getchar();
                entity_index =titles.vocab[title_index].entity_index[c];
            }
            else if(titles.vocab[title_index].entity_num>0)
                entity_index=titles.vocab[title_index].entity_index[0];
            
            for (a = 0; a < layer_size; a++) title_vec[a] = 0;
            printf("\nEntity: %s  Position in vocabulary: %d\n", entities.vocab[entity_index].item, entity_index);
            for (a = 0; a < layer_size; a++) title_vec[a] += titles.et_vec[a + entity_index * layer_size];
            FindNearest(N, title_vec);

            if(titles.vocab[title_index].sense_num>0){
                printf("There are %d out of KB senses!\n",titles.vocab[title_index].sense_num);
                for(b=0;b<titles.vocab[title_index].sense_num;b++){
                    for (a = 0; a < layer_size; a++) title_vec[a] = 0;
                    for (a = 0; a < layer_size; a++) title_vec[a] += titles.vec[a + b*layer_size+ title_index *max_sense_num * layer_size];
                    FindNearest(N, title_vec);
                }
            }
        }
        
    }
    
    return 0;
}
