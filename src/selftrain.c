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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 1100          //>longest entity + longest mention + 1
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define TEXT_MODEL "Text Model"
#define KG_MODEL "KG Model"
#define JOINT_MODEL "Joint Model"
#define MAX_MENTION 135

typedef float real;                    // Precision of float numbers

struct anchor_item {
    long long start_pos, length, index;
    int is_entity;
    char item[MAX_STRING];
};

struct vocab_item {
    long long cn, title_index;
    char *item;
};

struct vocab_title {
    //sense_num for NIL, entity_num for KB
    int sense_num, entity_num, max_entity_index_num;
    int *entity_index;
    char *item;
    long long cn;
};

struct {
    char map_file[MAX_STRING];
    int *vocab_hash, vocab_hash_size;
    int vocab_size;
    int *title_index;
    char *item;
}mapping_title;

struct model_var {
    char train_file[MAX_STRING], output_file[MAX_STRING], read_vocab_file[MAX_STRING], read_vector_file[MAX_STRING];
    struct vocab_item *vocab;
    int *vocab_hash, vocab_hash_size;
    long long vocab_max_size, vocab_size;
    long long train_items, item_count_actual, file_size;
    real starting_alpha, alpha;
    real *syn0, *syn1neg;
    char name[MAX_STRING];
    int *table;
}text_model, kg_model;

struct model_var2 {
    char train_file[MAX_STRING], output_file[MAX_STRING], read_vocab_file[MAX_STRING], read_vector_file[MAX_STRING];
    long long train_items, item_count_actual, file_size, anchor_count;
    real starting_alpha, alpha;
    char name[MAX_STRING];
    //for titles
    struct vocab_title *vocab;
    int *vocab_hash, *cluster_size, *et_cluster_size;
    int vocab_hash_size;
    long long vocab_max_size, vocab_size;
    real *syn0;
    real *mu;
    //for the sense linking to KB
    real *et_syn0;
    real *et_mu;
}joint_model;


int local_iter=0, binary = 1, debug_mode = 2, window = 5, min_count = 0, num_threads = 12, min_reduce = 1, max_sense_num = 10, save_iter = 1;
long long layer1_size = 200;
long long iter = 5, tmp_anchor_num = 0;
real alpha = 0.025, sample = 1e-3, cluster_threshold = -0.5;
real *expTable;
clock_t start;
char output_path[MAX_STRING];
int negative = 5;
const int table_size = 1e8;
const char save_interval = '\t';

void InitTextUnigramTable() {
    int a, i;
    double train_items_pow = 0;
    double d1, power = 0.75;
    text_model.table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < text_model.vocab_size; a++) train_items_pow += pow(text_model.vocab[a].cn, power);
    i = 0;
    d1 = pow(text_model.vocab[i].cn, power) / train_items_pow;
    for (a = 0; a < table_size; a++) {
        text_model.table[a] = i;
        if (a / (double)table_size > d1) {
            i++;
            d1 += pow(text_model.vocab[i].cn, power) / train_items_pow;
        }
        if (i >= text_model.vocab_size) i = text_model.vocab_size - 1;
    }
}

void InitKgUnigramTable() {
    int a, i;
    double train_items_pow = 0;
    double d1, power = 0.75;
    kg_model.table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < kg_model.vocab_size; a++) train_items_pow += pow(kg_model.vocab[a].cn, power);
    i = 0;
    d1 = pow(kg_model.vocab[i].cn, power) / train_items_pow;
    for (a = 0; a < table_size; a++) {
        kg_model.table[a] = i;
        if (a / (double)table_size > d1) {
            i++;
            d1 += pow(kg_model.vocab[i].cn, power) / train_items_pow;
        }
        if (i >= kg_model.vocab_size) i = kg_model.vocab_size - 1;
    }
}
//return the num of mention words
//split_pos indicates word start position in mention, range from 0 to mention length -1
int SplitMention(int *split_pos, char *mention) {
    int a = 0, b = 0, ch;
    if(mention[b]!=0){
        split_pos[a] = 0;
        a++;
        b++;
    }
    while (mention[b]!=0) {
        ch = mention[b];
        b++;
        if (ch == ' ')
            if(mention[b]!=0){
                split_pos[a] = b;
                a++;
                if(a>=MAX_MENTION){
                    printf("error! anchor's mention length is larger than %d!\n", MAX_MENTION);
                    printf("%s\n",mention);
                    break;
                }
            }
    }
    return a;
}

//return negative if that's not anchor, the int indicates how many words should be offset.
//return the word's start pos
int ReadAnchor(char *item, FILE *fin){
    int a = 0, ch, is_anchor = 0, prev_brace = 0, post_brace = 0, word_num=0, anchor_pos=0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if( prev_brace==0 ){
            if(ch == '['){
                prev_brace = 1;
                continue;
            }
            else{
                a=0;
                is_anchor --;
                break;
            }
        }
        is_anchor--;
        if(ch == ' ') word_num ++;
        if (ch == '\n' || word_num>MAX_MENTION) {
            a = 0;
            break;
        }
        if (ch == '|')
            anchor_pos = a+1;
        if ( (ch == ']') && (post_brace==0)){
            post_brace = 1;
            continue;
        }
        if(post_brace==1){
            if(ch == ']')
                is_anchor = anchor_pos;
            break;
        }
        item[a] = ch;
        a++;
        if (a >= MAX_STRING - 1){printf("error! too long string:\n %s\n",item); a--;break;}
    }
    item[a] = 0;
    return is_anchor;
}


//return negative if that's not anchor, the int indicates the offset.
//return 0 if read correctly
int ReadMention(char *item, FILE *fin){
    int a = 0, ch, is_anchor = 0, prev_brace = 0, post_brace = 0, word_num=0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if( prev_brace==0 ){
            if(ch == '{'){
                prev_brace = 1;
                continue;
            }
            else{
                a=0;
                is_anchor --;
                break;
            }
        }
        is_anchor--;
        if(ch == ' ') word_num ++;
        if (ch == '\n' || word_num>MAX_MENTION) {
            a = 0;
            break;
        }
        if ( (ch == '}') && (post_brace==0)){
            post_brace = 1;
            continue;
        }
        if(post_brace==1){
            if(ch == '}')
                is_anchor = 0;
            break;
        }
        item[a] = ch;
        a++;
        if (a >= MAX_STRING - 1){printf("error! too long string:\n %s\n",item); a--;break;}
    }
    item[a] = 0;
    return is_anchor;
}

// Reads a single word or anchor from a file, assuming space + tab + EOL to be word boundaries, [[anchor]]
//return -1 if read a word, 0 if [[anchor]], '|''s pos(1-length) if [[entity|mention]]
//return -2 if read a mention
int ReadText(char *item, FILE *fin) {
    int a = 0, ch, is_anchor=-1;
    
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n') || (ch == ']') || (ch == '[') || (ch == '{') || (ch == '}')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                if (ch == '[') ungetc(ch, fin);
                if (ch == '{') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(item, (char *)"</s>");
                return -1;
            }
            else if (ch == '['){
                is_anchor = ReadAnchor(item, fin);
                if (is_anchor<0){
                    fseek(fin, is_anchor, SEEK_CUR);
                    continue;
                }
                else return is_anchor;
            }
            else if(ch == '{'){
                is_anchor = ReadMention(item, fin);
                if (is_anchor<0){
                    fseek(fin, is_anchor, SEEK_CUR);
                    continue;
                }
                else return -2;
            }
            else continue;
        }
        item[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    item[a] = 0;
    return -1;
}

void ReadItem(char *item, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(item, (char *)"</s>");
                return;
            } else continue;
        }
        item[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    item[a] = 0;
}


int GetHash(char *item, long long size) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(item); a++) hash = hash * 257 + item[a];
    hash = hash % size;
    return hash;
}


// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchTextVocab(char *word) {
    unsigned int hash = GetHash(word, text_model.vocab_hash_size);
    while (1) {
        if (text_model.vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, text_model.vocab[text_model.vocab_hash[hash]].item)) return text_model.vocab_hash[hash];
        hash = (hash + 1) % text_model.vocab_hash_size;
    }
    return -1;
}

// Returns position of a title in the joint model vocab; if the title is not found, returns -1
int SearchTitleVocab(char *title) {
    unsigned int hash = GetHash(title, joint_model.vocab_hash_size);
    while (1) {
        if (joint_model.vocab_hash[hash] == -1) return -1;
        if (!strcmp(title, joint_model.vocab[joint_model.vocab_hash[hash]].item)) return joint_model.vocab_hash[hash];
        hash = (hash + 1) % joint_model.vocab_hash_size;
    }
    return -1;
}

// Returns position of an entity in the vocabulary; if the entity is not found, returns -1
int SearchKgVocab(char *entity) {
    unsigned int hash = GetHash(entity, kg_model.vocab_hash_size);
    while (1) {
        if (kg_model.vocab_hash[hash] == -1) return -1;
        if (!strcmp(entity, kg_model.vocab[kg_model.vocab_hash[hash]].item)) return kg_model.vocab_hash[hash];
        hash = (hash + 1) % kg_model.vocab_hash_size;
    }
    return -1;
}

int SearchMapVocab(char *map_title) {
    unsigned int hash = GetHash(map_title, mapping_title.vocab_hash_size);
    while (1) {
        if (mapping_title.vocab_hash[hash] == -1) return -1;
        if (!strcmp(map_title, &mapping_title.item[mapping_title.vocab_hash[hash]*MAX_STRING])) return mapping_title.vocab_hash[hash];
        hash = (hash + 1) % mapping_title.vocab_hash_size;
    }
    return -1;
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    text_model.vocab[text_model.vocab_size].item = (char *)calloc(length, sizeof(char));
    strcpy(text_model.vocab[text_model.vocab_size].item, word);
    text_model.vocab[text_model.vocab_size].cn = 0;
    text_model.vocab_size++;
    // Reallocate memory if needed
    if (text_model.vocab_size + 2 >= text_model.vocab_max_size) {
        text_model.vocab_max_size += 100000;
        text_model.vocab = (struct vocab_item *)realloc(text_model.vocab, text_model.vocab_max_size * sizeof(struct vocab_item));
    }
    hash = GetHash(word, text_model.vocab_hash_size);
    while (text_model.vocab_hash[hash] != -1) hash = (hash + 1) % text_model.vocab_hash_size;
    text_model.vocab_hash[hash] = text_model.vocab_size - 1;
    return text_model.vocab_size - 1;
}

void AddEntityIndexToTitle(int title_index, int entity_index){
    joint_model.vocab[title_index].entity_index[joint_model.vocab[title_index].entity_num] = entity_index;
    joint_model.vocab[title_index].entity_num ++;
    // Reallocate memory if needed
    if (joint_model.vocab[title_index].entity_num + 2 >= joint_model.vocab[title_index].max_entity_index_num) {
        joint_model.vocab[title_index].max_entity_index_num += 10;
        joint_model.vocab[title_index].entity_index = (int *)realloc(joint_model.vocab[title_index].entity_index, joint_model.vocab[title_index].max_entity_index_num * sizeof(int));
    }
}

// Adds a word to the vocabulary
int AddTitleToVocab(char *title) {
    //add item string
    unsigned int hash, length = strlen(title) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    joint_model.vocab[joint_model.vocab_size].item = (char *)calloc(length, sizeof(char));
    strcpy(joint_model.vocab[joint_model.vocab_size].item, title);
    //reset entitynum, sensenum, max entity index num
    joint_model.vocab[joint_model.vocab_size].entity_num = 0;
    joint_model.vocab[joint_model.vocab_size].sense_num = 0;
    joint_model.vocab[joint_model.vocab_size].max_entity_index_num = 10;
    joint_model.vocab[joint_model.vocab_size].cn = 0;
    //allocate memory for entity index
    joint_model.vocab[joint_model.vocab_size].entity_index = (int *)calloc(joint_model.vocab[joint_model.vocab_size].max_entity_index_num, sizeof(int));
    
    joint_model.vocab_size++;
    // Reallocate memory if needed
    if (joint_model.vocab_size + 2 >= joint_model.vocab_max_size) {
        joint_model.vocab_max_size += 100000;
        joint_model.vocab = (struct vocab_title *)realloc(joint_model.vocab, joint_model.vocab_max_size * sizeof(struct vocab_title));
    }
    hash = GetHash(title, joint_model.vocab_hash_size);
    while (joint_model.vocab_hash[hash] != -1) hash = (hash + 1) % joint_model.vocab_hash_size;
    joint_model.vocab_hash[hash] = joint_model.vocab_size - 1;
    
    return joint_model.vocab_size - 1;
}

// Adds an entity to the vocabulary
int AddEntityToVocab(char *entity) {
    unsigned int hash, length = strlen(entity) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    kg_model.vocab[kg_model.vocab_size].item = (char *)calloc(length, sizeof(char));
    strcpy(kg_model.vocab[kg_model.vocab_size].item, entity);
    kg_model.vocab[kg_model.vocab_size].cn = 0;
    kg_model.vocab_size++;
    // Reallocate memory if needed
    if (kg_model.vocab_size + 2 >= kg_model.vocab_max_size) {
        kg_model.vocab_max_size += 100000;
        kg_model.vocab = (struct vocab_item *)realloc(kg_model.vocab, kg_model.vocab_max_size * sizeof(struct vocab_item));
    }
    hash = GetHash(entity, kg_model.vocab_hash_size);
    while (kg_model.vocab_hash[hash] != -1) hash = (hash + 1) % kg_model.vocab_hash_size;
    kg_model.vocab_hash[hash] = kg_model.vocab_size - 1;
    return kg_model.vocab_size - 1;
}

// Used later for sorting by item counts
int VocabCompare( const void *a, const void *b) {
    return ((struct vocab_item *)b)->cn - ((struct vocab_item *)a)->cn;
}

// Used later for sorting by item counts
int VocabCompareTitle( const void *a, const void *b) {
    return ((struct vocab_title *)b)->cn - ((struct vocab_title *)a)->cn;
}

// Sorts the text vocabulary by frequency using word counts
void SortTextVocab() {
    int a;
    long long size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&text_model.vocab[1], text_model.vocab_size - 1, sizeof(struct vocab_item), VocabCompare);
    for (a = 0; a < text_model.vocab_hash_size; a++) text_model.vocab_hash[a] = -1;
    size = text_model.vocab_size;
    text_model.train_items = 0;
    for (a = 0; a < size; a++) {
        // items occuring less than min_count times will be discarded from the vocab
        if ((text_model.vocab[a].cn < min_count) && (a != 0)) {
            text_model.vocab_size--;
            free(text_model.vocab[a].item);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetHash(text_model.vocab[a].item, text_model.vocab_hash_size);
            while (text_model.vocab_hash[hash] != -1) hash = (hash + 1) % text_model.vocab_hash_size;
            text_model.vocab_hash[hash] = a;
            text_model.train_items += text_model.vocab[a].cn;
        }
    }
    text_model.vocab = (struct vocab_item *)realloc(text_model.vocab, (text_model.vocab_size + 1) * sizeof(struct vocab_item));
}

// Sorts the vocabulary by frequency using word counts
void SortKgVocab() {
    int a;
    long long size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&kg_model.vocab[1], kg_model.vocab_size - 1, sizeof(struct vocab_item), VocabCompare);
    for (a = 0; a < kg_model.vocab_hash_size; a++) kg_model.vocab_hash[a] = -1;
    size = kg_model.vocab_size;
    kg_model.train_items = 0;
    for (a = 0; a < size; a++) {
        // items occuring less than min_count times will be discarded from the vocab
        if ((kg_model.vocab[a].cn < min_count) && (a != 0)) {
            kg_model.vocab_size--;
            free(kg_model.vocab[a].item);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetHash(kg_model.vocab[a].item, kg_model.vocab_hash_size);
            while (kg_model.vocab_hash[hash] != -1) hash = (hash + 1) % kg_model.vocab_hash_size;
            kg_model.vocab_hash[hash] = a;
            kg_model.train_items += kg_model.vocab[a].cn;
        }
    }
    kg_model.vocab = (struct vocab_item *)realloc(kg_model.vocab, (kg_model.vocab_size + 1) * sizeof(struct vocab_item));
}

// Sorts the vocabulary by frequency using word counts
void SortTitleVocab() {
    int a;
    long long size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(joint_model.vocab, joint_model.vocab_size, sizeof(struct vocab_title), VocabCompareTitle);
    for (a = 0; a < joint_model.vocab_hash_size; a++) joint_model.vocab_hash[a] = -1;
    size = joint_model.vocab_size;
    joint_model.anchor_count = 0;
    for (a = 0; a < size; a++) {
        // items occuring less than min_count times will be discarded from the vocab
        if ((joint_model.vocab[a].cn < min_count) && (a != 0)) {
            joint_model.vocab_size--;
            free(joint_model.vocab[a].item);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetHash(joint_model.vocab[a].item, joint_model.vocab_hash_size);
            while (joint_model.vocab_hash[hash] != -1) hash = (hash + 1) % joint_model.vocab_hash_size;
            joint_model.vocab_hash[hash] = a;
            joint_model.anchor_count += joint_model.vocab[a].cn;
        }
    }
    joint_model.vocab = (struct vocab_title *)realloc(joint_model.vocab, joint_model.vocab_size * sizeof(struct vocab_title));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceTextVocab() {
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < text_model.vocab_size; a++) if (text_model.vocab[a].cn > min_reduce) {
        text_model.vocab[b].cn = text_model.vocab[a].cn;
        text_model.vocab[b].item = text_model.vocab[a].item;
        b++;
    } else free(text_model.vocab[a].item);
    text_model.vocab_size = b;
    for (a = 0; a < text_model.vocab_hash_size; a++) text_model.vocab_hash[a] = -1;
    for (a = 0; a < text_model.vocab_size; a++) {
        // Hash will be re-computed, as it is not actual
        hash = GetHash(text_model.vocab[a].item, text_model.vocab_hash_size);
        while (text_model.vocab_hash[hash] != -1) hash = (hash + 1) % text_model.vocab_hash_size;
        text_model.vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceKgVocab() {
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < kg_model.vocab_size; a++) if (kg_model.vocab[a].cn > min_reduce) {
        kg_model.vocab[b].cn = kg_model.vocab[a].cn;
        kg_model.vocab[b].item = kg_model.vocab[a].item;
        b++;
    } else free(kg_model.vocab[a].item);
    kg_model.vocab_size = b;
    for (a = 0; a < kg_model.vocab_hash_size; a++) kg_model.vocab_hash[a] = -1;
    for (a = 0; a < kg_model.vocab_size; a++) {
        // Hash will be re-computed, as it is not actual
        hash = GetHash(kg_model.vocab[a].item, kg_model.vocab_hash_size);
        while (kg_model.vocab_hash[hash] != -1) hash = (hash + 1) % kg_model.vocab_hash_size;
        kg_model.vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

void ReadTextVocab() {
    long long a, i = 0;
    char c;
    char word[MAX_STRING];
    FILE *fin = fopen(text_model.read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Text Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < text_model.vocab_hash_size; a++) text_model.vocab_hash[a] = -1;
    text_model.vocab_size = 0;
    while (1) {
        ReadItem(word, fin);
        if (feof(fin)) break;
        a = AddWordToVocab(word);
        fscanf(fin, "%lld%c", &text_model.vocab[a].cn, &c);
        i++;
    }
    SortTextVocab();
    if (debug_mode > 0) {
        printf("Text vocab size: %lld\n", text_model.vocab_size);
        printf("Words in train file: %lld\n", text_model.train_items);
    }
    fin = fopen(text_model.train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    text_model.file_size = ftell(fin);
    fclose(fin);
}

void ReadKgVocab() {
    long long a, i = 0;
    char c;
    char entity[MAX_STRING];
    FILE *fin = fopen(kg_model.read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < kg_model.vocab_hash_size; a++) kg_model.vocab_hash[a] = -1;
    kg_model.vocab_size = 0;
    while (1) {
        ReadItem(entity, fin);
        if (feof(fin)) break;
        a = AddEntityToVocab(entity);
        fscanf(fin, "%lld%c", &kg_model.vocab[a].cn, &c);
        i++;
    }
    SortKgVocab();
    if (debug_mode > 0) {
        printf("Entity Vocab size: %lld\n", kg_model.vocab_size);
        printf("entity in train file: %lld\n", kg_model.train_items);
    }
}

// return the pos of the left brace, -1 indicate there isn't.
int lengthOfTitle(char *entity_title){
    int i;
    int length = strlen(entity_title);
    for (i=0;i<length;i++)
        if (entity_title[i] == '('){
            if (i>0 && entity_title[i-1] == ' ') i--;
            break;
        }
    length = i;
    if (length >= MAX_STRING) length = MAX_STRING-1;
    return length;
}

void ReadTitleVocab() {
    long long a, i = 0, length=0;
    char c;
    char title[MAX_STRING], tmp_title[MAX_STRING];
    FILE *fin = fopen(joint_model.read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < joint_model.vocab_hash_size; a++) joint_model.vocab_hash[a] = -1;
    joint_model.vocab_size = 0;
    while (1) {
        ReadItem(title, fin);
        if (feof(fin)) break;
        // extract entity title without braces by iterating all the entity vocab
        length = lengthOfTitle(title);
        strncpy(tmp_title, title, sizeof(char)*length);
        tmp_title[length] = 0;
        i = SearchTitleVocab(tmp_title);
        if(i==-1) i = AddTitleToVocab(tmp_title);
        fscanf(fin, "%lld%c", &a, &c);
        joint_model.vocab[i].cn += a;
    }
    SortTitleVocab();
    if (debug_mode > 0) {
        printf("Title Vocab size: %lld\n", joint_model.vocab_size);
        printf("Anchors in train file: %lld\n", joint_model.anchor_count);
    }
}

void ReadMappingTitle(){
    long long a, i = 0, count = 0;
    char title[MAX_STRING], map_title[MAX_STRING], c;
    unsigned int hash;
    
    //init
    mapping_title.vocab_hash_size = 30000000;  // Maximum items in the vocabulary 30m*0.7=21m
    mapping_title.vocab_hash = (int *)calloc(mapping_title.vocab_hash_size, sizeof(int));
    
    for (a = 0; a < mapping_title.vocab_hash_size; a++) mapping_title.vocab_hash[a] = -1;
    
    FILE *fin = fopen(mapping_title.map_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    //init
    fscanf(fin, "%d%c", &mapping_title.vocab_size, &c);
    for (a = 0; a < mapping_title.vocab_hash_size; a++) mapping_title.vocab_hash[a] = -1;
    mapping_title.item = (char *)malloc(mapping_title.vocab_size * MAX_STRING * sizeof(char));
    mapping_title.title_index = (int *)calloc(mapping_title.vocab_size, sizeof(int));
    a = 0;
    while (1) {
        ReadItem(map_title, fin);
        ReadItem(title, fin);
        if (feof(fin)) break;
        strcpy(&mapping_title.item[count*MAX_STRING], map_title);
        mapping_title.item[strlen(map_title)] = 0;
        i = SearchTitleVocab(title);
        mapping_title.title_index[count++] = i;
        if (i!=-1 && joint_model.vocab[i].entity_num > 0) a++;
        hash = GetHash(map_title, mapping_title.vocab_hash_size);
        while (mapping_title.vocab_hash[hash] != -1) hash = (hash + 1) % mapping_title.vocab_hash_size;
        mapping_title.vocab_hash[hash] = count-1;
        fscanf(fin, "%c", &c);
    }
    
    if (debug_mode > 0) {
        printf("Mapped Title Vocab size: %d, cover %lld titles.\n", mapping_title.vocab_size, a);
    }
}

void initTitleVocab(){
    long long a, i;
    int length=0, cover_entity_num = 0;
    char tmp_title[MAX_STRING];
    
    for (a=0;a<kg_model.vocab_size;a++){
        // extract entity title without braces by iterating all the entity vocab
        length = lengthOfTitle(kg_model.vocab[a].item);
        strncpy(tmp_title, kg_model.vocab[a].item, sizeof(char)*length);
        tmp_title[length] = 0;
        
        i = SearchTitleVocab(tmp_title);
        if(i!=-1) {
            AddEntityIndexToTitle(i, a);
            cover_entity_num ++;
        }
        kg_model.vocab[a].title_index = i;
    }
    printf("cover %d entities!\n", cover_entity_num);
}

void InitTextNet() {
    long long a, b;
    a = posix_memalign((void **)&(text_model.syn0), 128, (long long)text_model.vocab_size * layer1_size * sizeof(real));
    if (text_model.syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    if (negative>0) {
        a = posix_memalign((void **)&(text_model.syn1neg), 128, (long long)text_model.vocab_size * layer1_size * sizeof(real));
        if (text_model.syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (a = 0; a < text_model.vocab_size; a++) for (b = 0; b < layer1_size; b++)
            text_model.syn1neg[a * layer1_size + b] = 0;
    }
}

void InitTitleNet() {
    long long a, b, c;
    unsigned long long next_random = 1;
    //title entity embedding for KB
    a = posix_memalign((void **)&(joint_model.et_syn0), 128, (long long)kg_model.vocab_size * layer1_size * sizeof(real));
    if (joint_model.et_syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    //title entity cluster
    a = posix_memalign((void **)&(joint_model.et_mu), 128, (long long)kg_model.vocab_size * layer1_size * sizeof(real));
    if (joint_model.et_mu == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < kg_model.vocab_size; a++) for (b = 0; b < layer1_size; b++)
        joint_model.et_mu[a * layer1_size + b] = 0;
    
    //title entity cluster size
    joint_model.et_cluster_size = (int *)calloc(kg_model.vocab_size, sizeof(int));
    
    //title sense embedding for NIL
    a = posix_memalign((void **)&(joint_model.syn0), 128, (long long)joint_model.vocab_size * max_sense_num * layer1_size * sizeof(real));
    if (joint_model.syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    //sense cluster center
    a = posix_memalign((void **)&(joint_model.mu), 128, (long long)joint_model.vocab_size * max_sense_num * layer1_size * sizeof(real));
    if (joint_model.mu == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < joint_model.vocab_size; a++) for (b = 0; b < layer1_size; b++)
        joint_model.mu[a * layer1_size + b] = 0;
    
    //sense cluster size
    joint_model.cluster_size = (int *)calloc(joint_model.vocab_size * max_sense_num, sizeof(int));
    
    //init title sense embedding
    for (a = 0; a < joint_model.vocab_size; a++) for (b = 0; b < max_sense_num; b++) for (c=0;c<layer1_size;c++){
        next_random = next_random * (unsigned long long)25214903917 + 11;
        joint_model.syn0[a * max_sense_num * layer1_size + b * layer1_size + c] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
}

void findMentionSense(struct anchor_item *anchor, real *vec, int title_index){
    int total_c_size = 0, a, b, closest = -1, et_closest = -1, c_size, tmp_entity_index;
    real dist = 0.0, closest_dist = -1.0, et_closest_dist = -1.0, len_v;
    if (title_index == -1){
        anchor->index = -1;
        return;
    }
    //norm of vec
    len_v = 0;
    for (a = 0; a < layer1_size; a++) len_v += vec[a] * vec[a];
    len_v = sqrt(len_v);
    for (a = 0; a < layer1_size; a++) vec[a] /= len_v;
    
    //choose in entity sense
    for (a = 0; a < joint_model.vocab[title_index].entity_num; a++){
        tmp_entity_index = joint_model.vocab[title_index].entity_index[a];
        c_size = joint_model.et_cluster_size[tmp_entity_index];
        if(c_size<=0)
            continue;
        total_c_size += c_size;
        //norm
        len_v = 0;
        for (b = 0; b < layer1_size; b++)
            len_v += (joint_model.et_mu[tmp_entity_index * layer1_size + b]/c_size) * (joint_model.et_mu[tmp_entity_index * layer1_size + b]/c_size);
        len_v = sqrt(len_v);
        //cosine similarity * cluster_size
        dist = 0;
        for (b = 0; b < layer1_size; b++)
            dist += vec[b] * (joint_model.et_mu[tmp_entity_index * layer1_size + b]/len_v);
        //
        if(dist > et_closest_dist){
            et_closest_dist = dist;
            et_closest = tmp_entity_index;
        }
    }
    
    //choose in cluster sense
    for (a = 0; a < joint_model.vocab[title_index].sense_num; a++){
        c_size = joint_model.cluster_size[title_index*max_sense_num+a];
        if(c_size<=0)
            continue;
        total_c_size += c_size;
        len_v = 0;
        for (b = 0; b < layer1_size; b++)
            len_v += (joint_model.mu[title_index * max_sense_num * layer1_size + a * layer1_size + b]/c_size) * (joint_model.mu[title_index * max_sense_num * layer1_size + a * layer1_size + b]/c_size);
        len_v = sqrt(len_v);
        //cosine similarity * cluster_size
        dist = 0;
        for (b = 0; b < layer1_size; b++)
            dist += vec[b] * (joint_model.mu[title_index * max_sense_num * layer1_size + a * layer1_size + b]/len_v);
        if(dist > closest_dist){
            closest_dist = dist;
            closest = a;
        }
    }
    if (total_c_size <= 0){
        anchor->index = -1;
        return;
    }
    et_closest_dist /= total_c_size;
    closest_dist /= total_c_size;
    if (et_closest_dist < cluster_threshold && closest_dist < cluster_threshold){
        anchor->is_entity = 0;
        //create a new sense
        if (joint_model.vocab[title_index].sense_num < max_sense_num){
            anchor->index = title_index * max_sense_num * layer1_size + joint_model.vocab[title_index].sense_num*layer1_size;
            //update cluster size & mu
            joint_model.cluster_size[title_index * max_sense_num + joint_model.vocab[title_index].sense_num] = 1;
            for (a = 0; a < layer1_size; a++)
                joint_model.mu[title_index * max_sense_num * layer1_size + joint_model.vocab[title_index].sense_num * layer1_size + a] = vec[a];
            joint_model.vocab[title_index].sense_num++;
        }
        else{
            anchor->index = -1;
        }
    }
    else if (et_closest_dist > closest_dist){
        //closer to entity sense
        anchor->is_entity = 1;
        anchor->index = et_closest * layer1_size;
        //update mu and cluster size
        joint_model.et_cluster_size[et_closest] ++;
        for (a = 0; a < layer1_size; a++)
            joint_model.et_mu[et_closest * layer1_size + a] += vec[a];
    }
    else{
        // closer to NIL sense
        anchor->is_entity = 0;
        // [0] of the title_index's nearest sense
        anchor->index = title_index * max_sense_num * layer1_size + closest * layer1_size;
        //update mu and cluster size
        joint_model.cluster_size[title_index * max_sense_num + closest] ++;
        for (a = 0; a < layer1_size; a++)
            joint_model.mu[title_index * max_sense_num * layer1_size + closest * layer1_size + a] += vec[a];
    }
}


void *TrainTextModelThread(void *id) {
    long long a, b, d, word=-1, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, anchor_count = 0, anchor_position=0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label = 0;
    unsigned long long next_random = (long long)id;
    int anchor_pos = -1, word_begin[MAX_MENTION], words_in_mention = 1, cw = 0, map_index = 0;
    char item[MAX_STRING], tmp_item[MAX_STRING], str_word[MAX_STRING];
    size_t tmp_item_len = 0;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    real *tmp_context_vec = (real *)calloc(layer1_size, sizeof(real));
    struct anchor_item *anchors = (struct anchor_item *)calloc(MAX_SENTENCE_LENGTH, sizeof(struct anchor_item));
    FILE *fi = fopen(text_model.train_file, "rb");
    fseek(fi, text_model.file_size / (long long)num_threads * (long long)id, SEEK_SET);
    for(a=0;a<MAX_MENTION;a++) word_begin[a] = 0;
    while (1) {
        if (word_count - last_word_count > 10000) {
            text_model.item_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%c%s: Alpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, text_model.name, text_model.alpha,
                       text_model.item_count_actual / (real)(text_model.train_items + 1) * 100,
                       text_model.item_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            text_model.alpha = text_model.starting_alpha * (1 - text_model.item_count_actual / (real)(iter * text_model.train_items + 1));
            if (text_model.alpha < text_model.starting_alpha * 0.0001) text_model.alpha = text_model.starting_alpha * 0.0001;
        }
        if (sentence_length == 0) {
            anchor_count = 0;
            while(1){
                anchor_pos = ReadText(item, fi);
                if (feof(fi)) break;
                if (anchor_pos>=0) anchors[anchor_count].start_pos = sentence_length;
                if(anchor_pos > 0){
                    tmp_item_len = strlen(item)-anchor_pos;
                    strncpy(tmp_item, &item[anchor_pos], sizeof(char)*tmp_item_len);
                }
                else{
                    tmp_item_len = strlen(item);
                    strncpy(tmp_item, item, sizeof(char)*tmp_item_len);
                }
                tmp_item[tmp_item_len] = 0;
                words_in_mention = 1;
                if (anchor_pos != -1){
                    words_in_mention = SplitMention(word_begin, tmp_item);
                }
                
                for(b=0;b<words_in_mention;b++){
                    if (anchor_pos == -1){
                        tmp_item_len = strlen(tmp_item);
                        strncpy(str_word, tmp_item, sizeof(char)*tmp_item_len);
                    }
                    else if(b+1<words_in_mention){
                        tmp_item_len = word_begin[b+1]-1-word_begin[b];
                        strncpy(str_word, &tmp_item[word_begin[b]], sizeof(char)*tmp_item_len);
                    }
                    else{
                        tmp_item_len = strlen(&tmp_item[word_begin[b]]);
                        strncpy(str_word, &tmp_item[word_begin[b]], sizeof(char)*tmp_item_len);
                    }
                    str_word[tmp_item_len]=0;
                    word = SearchTextVocab(str_word);
                    word_count++;
                    if (word == -1) continue;
                    if (word == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        real ran = (sqrt(text_model.vocab[word].cn / (sample * text_model.train_items)) + 1) * (sample * text_model.train_items) / text_model.vocab[word].cn;
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                    }
                    sen[sentence_length] = word;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                }
                if((anchor_pos == -2 || anchor_pos>=0) && b >= words_in_mention-1){
                    anchors[anchor_count].length = sentence_length - anchors[anchor_count].start_pos;
                    if(anchors[anchor_count].length>0){
                        if (anchor_pos >= 0){
                            anchors[anchor_count].is_entity = 1;
                            if (anchor_pos == 0)
                                strcpy(anchors[anchor_count].item, item);
                            else{
                                strncpy(anchors[anchor_count].item, item, anchor_pos-1);
                                anchors[anchor_count].item[anchor_pos-1] = 0;
                            }
                            anchors[anchor_count].index = SearchKgVocab(anchors[anchor_count].item);
                        }
                        else{
                            anchors[anchor_count].is_entity = 0;
                            strcpy(anchors[anchor_count].item, item);
                            anchors[anchor_count].index = -1;
                        }
                        
                        if(anchors[anchor_count].is_entity == 0 || ( anchors[anchor_count].is_entity==1 && anchors[anchor_count].index!=-1 && kg_model.vocab[anchors[anchor_count].index].title_index != -1)){
                            anchor_count++;
                            if(anchor_count >= MAX_SENTENCE_LENGTH) anchor_count--;
                        }
                    }
                }
                if(word == 0 || sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (feof(fi) || (word_count > text_model.train_items / num_threads)) {
            text_model.item_count_actual += word_count - last_word_count;
            break;
        }
        
        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
        //train skip-gram
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
            c = sentence_position - window + a;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;
            
            l1 = last_word * layer1_size;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
            // NEGATIVE SAMPLING
            if (negative > 0) for (d = 0; d < negative + 1; d++) {
                if (d == 0) {
                    target = word;
                    label = 1;
                } else {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = text_model.table[(next_random >> 16) % table_size];
                    if (target == 0) target = next_random % (text_model.vocab_size - 1) + 1;
                    if (target == word) continue;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++) f += text_model.syn0[c + l1] * text_model.syn1neg[c + l2];
                if (f > MAX_EXP) g = (label - 1) * text_model.alpha;
                else if (f < -MAX_EXP) g = (label - 0) * text_model.alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * text_model.alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * text_model.syn1neg[c + l2];
                for (c = 0; c < layer1_size; c++) text_model.syn1neg[c + l2] += g * text_model.syn0[c + l1];
            }
            // Learn weights input -> hidden
            for (c = 0; c < layer1_size; c++) text_model.syn0[c + l1] += neu1e[c];
        }
        
        sentence_position++;
        if (sentence_position >= sentence_length) {
            // train word embedding finished! start update title sense embedding ...
            if (anchor_count > 0){
                
                for(anchor_position=0;anchor_position<anchor_count;anchor_position++){
                    //choose sense for mention NIL
                    for (c = 0; c < layer1_size; c++) tmp_context_vec[c] = 0;
                    cw = 0;
                    for (a = b; a < window * 2 + 1 - b; a++)
                        if(a == window)
                            sentence_position = anchors[anchor_position].start_pos + anchors[anchor_position].length-1;
                        else {
                            c = sentence_position - window + a;
                            if (c < 0) continue;
                            if (c >= sentence_length) continue;
                            last_word = sen[c];
                            if (last_word == -1) continue;
                            for (c = 0; c < layer1_size; c++)
                                //current context vec
                                tmp_context_vec[c] += text_model.syn0[c + last_word * layer1_size];
                            cw++;
                        }
                    if (cw <= 0) continue;
                    //choose sense
                    if (anchors[anchor_position].is_entity == 0){
                        map_index = SearchMapVocab(anchors[anchor_position].item);
                        if (map_index != -1)
                            findMentionSense(&anchors[anchor_position], tmp_context_vec, mapping_title.title_index[map_index]);
                        if(anchors[anchor_count].index == -1)
                            continue;
                    }
                    tmp_anchor_num++;
                    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
                    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    b = next_random % window;
                    sentence_position = anchors[anchor_position].start_pos;
                    
                    l1 = anchors[anchor_count].index;
                    
                    //train skip-gram
                    for (a = b; a < window * 2 + 1 - b; a++)
                        if(a == window)
                            sentence_position = anchors[anchor_position].start_pos + anchors[anchor_position].length-1;
                        else {
                            c = sentence_position - window + a;
                            if (c < 0) continue;
                            if (c >= sentence_length) continue;
                            last_word = sen[c];
                            if (last_word == -1) continue;
                            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                            // NEGATIVE SAMPLING
                            if (negative > 0) for (d = 0; d < negative + 1; d++) {
                                if (d == 0) {
                                    target = last_word;
                                    label = 1;
                                } else {
                                    next_random = next_random * (unsigned long long)25214903917 + 11;
                                    target = text_model.table[(next_random >> 16) % table_size];
                                    if (target == 0) target = next_random % (text_model.vocab_size - 1) + 1;
                                    if (target == last_word) continue;
                                    label = 0;
                                }
                                l2 = target * layer1_size;
                                f = 0;
                                if (anchors[anchor_count].is_entity==1)
                                    for (c = 0; c < layer1_size; c++) f += joint_model.et_syn0[c + l1] * text_model.syn1neg[c + l2];
                                else
                                    for (c = 0; c < layer1_size; c++) f += joint_model.syn0[c + l1] * text_model.syn1neg[c + l2];
                                if (f > MAX_EXP) g = (label - 1) * joint_model.alpha;
                                else if (f < -MAX_EXP) g = (label - 0) * joint_model.alpha;
                                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * joint_model.alpha;
                                for (c = 0; c < layer1_size; c++) neu1e[c] += g * text_model.syn1neg[c + l2];
                                if (anchors[anchor_count].is_entity==1)
                                    for (c = 0; c < layer1_size; c++) text_model.syn1neg[c + l2] += g * joint_model.et_syn0[c + l1];
                                else
                                    for (c = 0; c < layer1_size; c++) text_model.syn1neg[c + l2] += g * joint_model.syn0[c + l1];
                            }
                            // Learn weights input -> hidden
                            if (anchors[anchor_count].is_entity==1)
                                for (c = 0; c < layer1_size; c++) joint_model.et_syn0[c + l1] += neu1e[c];
                            else
                                for (c = 0; c < layer1_size; c++) joint_model.syn0[c + l1] += neu1e[c];
                        }
                }
            }
            
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void TrainTextModel(){
    long a, b, c;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    real len;
    
    start = clock();
    printf("\nStarting training text model using file %s\n", text_model.train_file);
    text_model.item_count_actual = 0;
    text_model.starting_alpha = text_model.alpha;
    tmp_anchor_num = 0;
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainTextModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("Total anchors in train file: %lld\n", tmp_anchor_num);
    
    //norm embedding of words and title, and reset the cluster
    for (a=0;a<text_model.vocab_size;a++){
        len = 0;
        for (b = 0; b < layer1_size; b++) len += text_model.syn0[b + a * layer1_size] * text_model.syn0[b + a * layer1_size];
        len = sqrt(len);
        for (b = 0; b < layer1_size; b++) text_model.syn0[b + a * layer1_size] /= len;
        // norm syn1neg
        len = 0;
        for (b = 0; b < layer1_size; b++) len += text_model.syn1neg[b + a * layer1_size] * text_model.syn1neg[b + a * layer1_size];
        len = sqrt(len);
        for (b = 0; b < layer1_size; b++) text_model.syn1neg[b + a * layer1_size] /= len;
    }
    
    //reset the context cluster center
    for (a = 0; a < joint_model.vocab_size; a++){
        //entity cluster
        for (b=0;b<joint_model.vocab[a].entity_num;b++){
            //norm title entity vector
            len = 0;
            for (c = 0; c < layer1_size; c++) len += joint_model.et_syn0[joint_model.vocab[a].entity_index[b] * layer1_size + c] * joint_model.et_syn0[joint_model.vocab[a].entity_index[b] * layer1_size + c];
            len = sqrt(len);
            for (c = 0; c < layer1_size; c++) joint_model.et_syn0[joint_model.vocab[a].entity_index[b] * layer1_size + c] /= len;
            //reset cluster
            if (joint_model.et_cluster_size[joint_model.vocab[a].entity_index[b]]>0){
                for (c=0;c<layer1_size;c++)
                    joint_model.et_mu[joint_model.vocab[a].entity_index[b]*layer1_size+c] /= joint_model.et_cluster_size[joint_model.vocab[a].entity_index[b]];
                joint_model.et_cluster_size[joint_model.vocab[a].entity_index[b]] = 1;
                //norm title entity cluster
                len = 0;
                for (c = 0; c < layer1_size; c++) len += joint_model.et_mu[joint_model.vocab[a].entity_index[b] * layer1_size + c] * joint_model.et_mu[joint_model.vocab[a].entity_index[b] * layer1_size + c];
                len = sqrt(len);
                for (c = 0; c < layer1_size; c++) joint_model.et_mu[joint_model.vocab[a].entity_index[b] * layer1_size + c] /= len;
                
            }
        }
        //NIL cluster
        for (b=0;b<joint_model.vocab[a].sense_num;b++){
            //norm for NIL sense vector
            len = 0;
            for (c = 0; c < layer1_size; c++) len += joint_model.syn0[a *max_sense_num* layer1_size + b * layer1_size + c] * joint_model.syn0[a *max_sense_num* layer1_size + b * layer1_size + c];
            len = sqrt(len);
            for (c = 0; c < layer1_size; c++) joint_model.syn0[a *max_sense_num* layer1_size + b * layer1_size + c] /= len;
            //reset cluster
            if (joint_model.cluster_size[a*max_sense_num+b]>0){
                for (c=0;c<layer1_size;c++)
                    joint_model.mu[a*max_sense_num*layer1_size + b*layer1_size+c] /= joint_model.cluster_size[a*max_sense_num+b];
                joint_model.cluster_size[a*max_sense_num+b] = 1;
                //norm for NIL context center
                len = 0;
                for (c = 0; c < layer1_size; c++) len += joint_model.mu[a *max_sense_num* layer1_size + b * layer1_size + c] * joint_model.mu[a *max_sense_num* layer1_size + b * layer1_size + c];
                len = sqrt(len);
                for (c = 0; c < layer1_size; c++) joint_model.mu[a *max_sense_num* layer1_size + b * layer1_size + c] /= len;
                
            }
        }
    }
}

void SaveWordVector(int id){
    char tmp_output_file[MAX_STRING];
    FILE *fo;
    long a, b;
    if(output_path[0]!=0){
        sprintf(tmp_output_file, "%s%d.dat", text_model.output_file, id);
        fo = fopen(tmp_output_file, "wb");
    }
    else{
        printf("output path doesn't exist!");
        return;
    }
    // Save the item vectors
    fprintf(fo, "%lld %lld\n", text_model.vocab_size, layer1_size);
    for (a = 0; a < text_model.vocab_size; a++) {
        fprintf(fo, "%s%c", text_model.vocab[a].item, save_interval);
        
        if (binary) for (b = 0; b < layer1_size; b++) fwrite(&(text_model.syn0[a * layer1_size + b]), sizeof(real), 1, fo);
        else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", text_model.syn0[a * layer1_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

void SaveTitleVector(int id){
    char tmp_output_file[MAX_STRING];
    FILE *fo;
    long a, b, c;
    if(output_path[0]!=0){
        sprintf(tmp_output_file, "%s%d.dat", joint_model.output_file, id);
        fo = fopen(tmp_output_file, "wb");
    }
    else{
        printf("output path doesn't exist!");
        return;
    }
    // Save the item vectors
    fprintf(fo, "%lld %lld %d\n", joint_model.vocab_size, layer1_size, max_sense_num);
    for (a = 0; a < joint_model.vocab_size; a++) {
        
        fprintf(fo, "%s%c%d%c%d%c", joint_model.vocab[a].item, save_interval, joint_model.vocab[a].sense_num, save_interval,joint_model.vocab[a].entity_num, save_interval);
        
        for (b = 0; b < joint_model.vocab[a].sense_num; b++){
            
            for (c = 0; c < layer1_size; c++)
                fwrite(&(joint_model.syn0[a *max_sense_num* layer1_size + b * layer1_size + c]), sizeof(real), 1, fo);
            
            //regularization before save
            if (joint_model.cluster_size[a *max_sense_num+b]>0)
                for (c = 0; c < layer1_size; c++)
                    joint_model.mu[a *max_sense_num* layer1_size + b * layer1_size + c] /= joint_model.cluster_size[a *max_sense_num+b];
            
            
            
            for (c = 0; c < layer1_size; c++)
                fwrite(&(joint_model.mu[a *max_sense_num* layer1_size + b * layer1_size + c]), sizeof(real), 1, fo);
        }
        fprintf(fo, "\n");
        for (b = 0; b < joint_model.vocab[a].entity_num; b++){
            fprintf(fo, "%s%c", kg_model.vocab[joint_model.vocab[a].entity_index[b]].item, save_interval);
            
            
            for (c = 0; c < layer1_size; c++)
                fwrite(&(joint_model.et_syn0[joint_model.vocab[a].entity_index[b] * layer1_size + c]), sizeof(real), 1, fo);
            // regularization before save
            if (joint_model.et_cluster_size[joint_model.vocab[a].entity_index[b]]>0)
                for (c = 0; c < layer1_size; c++)
                    joint_model.et_mu[joint_model.vocab[a].entity_index[b] * layer1_size + c] /= joint_model.et_cluster_size[joint_model.vocab[a].entity_index[b]];
            
            for (c = 0; c < layer1_size; c++)
                fwrite(&(joint_model.et_mu[joint_model.vocab[a].entity_index[b] * layer1_size + c]), sizeof(real), 1, fo);
            fprintf(fo, "\n");
        }
        
    }
    fclose(fo);
}

void LoadWordVector(){
    FILE *f;
    long long a, b, word_index, vocab_size, layer_size;
    char tmp_word[MAX_STRING];
    real *tmp_syn0;
    int load_vec_count = 0;
    char d;
    f = fopen(text_model.read_vector_file, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return;
    }
    fscanf(f, "%lld", &vocab_size);
    fscanf(f, "%lld", &layer_size);
    fscanf(f, "%c", &d);
    if(layer_size!=layer1_size){
        printf("word vec size doesn't match!\n");
        return;
    }
    tmp_syn0 = (real *)calloc(layer1_size, sizeof(real));
    for (b = 0; b < vocab_size; b++) {
        a = 0;
        while (1) {
            tmp_word[a] = fgetc(f);
            if (feof(f) || (tmp_word[a] == '\t')) break;
            if ((a < MAX_STRING) && (tmp_word[a] != '\n')) a++;
        }
        tmp_word[a] = 0;
        word_index = SearchTextVocab(tmp_word);
        if (word_index!=-1){
            for (a = 0; a < layer1_size; a++) fread(&text_model.syn0[a + word_index * layer1_size], sizeof(float), 1, f);
            load_vec_count ++;
        }
        else
            for (a = 0; a < layer1_size; a++) fread(tmp_syn0, sizeof(float), 1, f);
    }
    fclose(f);
    printf("successfully load %d word vectors!\n", load_vec_count);
}

void LoadTitleVector(){
    FILE *f;
    long long a, b, c, e, title_index, entity_index, vocab_size;
    int sense_num = 0, entity_num = 0, layer_size, tmp_max_sense_num;
    char tmp_title[MAX_STRING], tmp_entity[MAX_STRING];
    real *tmp_syn0, len = 0;
    char d;
    
    f = fopen(joint_model.read_vector_file, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return;
    }
    fscanf(f, "%lld", &vocab_size);
    fscanf(f, "%d", &layer_size);
    if (layer1_size != layer_size){
        printf("unmatched title vector dimension!\n");
        return;
    }
    fscanf(f, "%d", &tmp_max_sense_num);
    fscanf(f, "%c", &d);
    if (tmp_max_sense_num != max_sense_num){
        printf("unmatched title vector max sense num!\n");
        return;
    }
    //count the num of loaded titles and entities
    tmp_max_sense_num = 0;
    layer_size = 0;
    tmp_syn0 = (real *)calloc(layer1_size, sizeof(real));
    for (b = 0; b < vocab_size; b++) {
        a = 0;
        while (1) {
            tmp_title[a] = fgetc(f);
            if (feof(f) || (tmp_title[a] == '\t')) break;
            if ((a < MAX_STRING) && (tmp_title[a] != '\n')) a++;
        }
        tmp_title[a] = 0;
        
        title_index = SearchTitleVocab(tmp_title);
        fscanf(f, "%d%c%d%c", &sense_num, &d, &entity_num, &d);
        if (title_index!=-1){
            tmp_max_sense_num ++;
            for (e = 0; e < sense_num; e++){
                joint_model.vocab[title_index].sense_num ++;
                for (c = 0; c < layer1_size; c++) fread(&joint_model.syn0[c + e * layer1_size + title_index * max_sense_num * layer1_size], sizeof(float), 1, f);
                //norm
                len = 0;
                for (c = 0; c < layer1_size; c++) len += joint_model.syn0[c + e * layer1_size + title_index * max_sense_num * layer1_size] * joint_model.syn0[c + e * layer1_size + title_index * max_sense_num * layer1_size];
                len = sqrt(len);
                for (c = 0; c < layer1_size; c++) joint_model.syn0[c + e * layer1_size + title_index * max_sense_num * layer1_size] /= len;
                
                for (c = 0; c < layer1_size; c++) fread(&joint_model.mu[c + e * layer1_size + title_index * max_sense_num * layer1_size], sizeof(float), 1, f);
                if(!isnan(joint_model.mu[e * layer1_size + title_index * max_sense_num * layer1_size])){
                    //norm
                    len = 0;
                    for (c = 0; c < layer1_size; c++) len += joint_model.mu[c + e * layer1_size + title_index * max_sense_num * layer1_size] * joint_model.mu[c + e * layer1_size + title_index * max_sense_num * layer1_size];
                    len = sqrt(len);
                    for (c = 0; c < layer1_size; c++) joint_model.mu[c + e * layer1_size + title_index * max_sense_num * layer1_size] /= len;
                    
                    joint_model.cluster_size[title_index * max_sense_num + e] = 1;
                }
                else
                    joint_model.cluster_size[title_index * max_sense_num + e] = 0;
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
                    layer_size++;
                    for (c = 0; c < layer1_size; c++) fread(&joint_model.et_syn0[c + entity_index * layer1_size], sizeof(float), 1, f);
                    //norm
                    len = 0;
                    for (c = 0; c < layer1_size; c++) len += joint_model.et_syn0[c + entity_index * layer1_size] * joint_model.et_syn0[c + entity_index * layer1_size];
                    len = sqrt(len);
                    for (c = 0; c < layer1_size; c++) joint_model.et_syn0[c + entity_index * layer1_size] /= len;
                    
                    for (c = 0; c < layer1_size; c++) fread(&joint_model.et_mu[c + entity_index * layer1_size], sizeof(float), 1, f);
                    if(!isnan(joint_model.mu[e * layer1_size + title_index * max_sense_num * layer1_size])){
                        //norm
                        len = 0;
                        for (c = 0; c < layer1_size; c++) len += joint_model.et_mu[c + entity_index * layer1_size] * joint_model.et_mu[c + entity_index * layer1_size];
                        len = sqrt(len);
                        for (c = 0; c < layer1_size; c++) joint_model.et_mu[c + entity_index * layer1_size] /= len;
                        
                        joint_model.et_cluster_size[entity_index] = 1;
                    }
                    else
                        joint_model.et_cluster_size[entity_index] = 0;
                }
                else{
                    for (c = 0; c < layer1_size; c++) fread(tmp_syn0, sizeof(float), 1, f);
                    for (c = 0; c < layer1_size; c++) fread(tmp_syn0, sizeof(float), 1, f);
                }
                fscanf(f, "%c", &d);
            }
        }// end for reading title vectors
        else{
            //skip this title
            for (e = 0; e < sense_num; e++){
                for (c = 0; c < layer1_size; c++) fread(tmp_syn0, sizeof(float), 1, f);
                for (c = 0; c < layer1_size; c++) fread(tmp_syn0, sizeof(float), 1, f);
            }
            fscanf(f, "%c", &d);
            for (e = 0; e < entity_num; e++){
                a = 0;
                while (1) {
                    tmp_entity[a] = fgetc(f);
                    if (feof(f) || (tmp_entity[a] == '\t')) break;
                    if ((a < MAX_STRING) && (tmp_entity[a] != '\n')) a++;
                }
                tmp_entity[a] = 0;
                for (c = 0; c < layer1_size; c++) fread(tmp_syn0, sizeof(float), 1, f);
                for (c = 0; c < layer1_size; c++) fread(tmp_syn0, sizeof(float), 1, f);
                fscanf(f, "%c", &d);
            }
        }
    }
    fclose(f);
    printf("successfully load %d title with %d sense vectors!\n", tmp_max_sense_num, layer_size);
}

void InitTextModel(){
    text_model.vocab_max_size = 2500000;      //vocab word size is 2.7m
    text_model.vocab_size = 0;
    text_model.train_items = 0;
    text_model.item_count_actual = 0;
    text_model.file_size = 0;
    text_model.alpha = alpha;
    text_model.vocab_hash_size = 30000000;  // Maximum items in the vocabulary 30m*0.7=21m
    
    text_model.vocab = (struct vocab_item *)calloc(text_model.vocab_max_size, sizeof(struct vocab_item));
    text_model.vocab_hash = (int *)calloc(text_model.vocab_hash_size, sizeof(int));
    
    if (text_model.read_vocab_file[0] != 0) {
        ReadTextVocab();
        InitTextNet();
        if(text_model.read_vector_file[0]!=0)
            LoadWordVector();
        if (negative > 0) InitTextUnigramTable();
    }
    else {
        printf("please init text model!");
        return;
    }
}

void InitKgModel(){
    kg_model.vocab_max_size = 5000000;      //vocab entity size is 5.13m
    kg_model.vocab_size = 0;
    kg_model.train_items = 0;
    kg_model.item_count_actual = 0;
    kg_model.file_size = 0;
    kg_model.alpha = alpha;
    kg_model.vocab_hash_size = 30000000;  // Maximum items in the vocabulary 30m*0.7=21m
    kg_model.vocab = (struct vocab_item *)calloc(kg_model.vocab_max_size, sizeof(struct vocab_item));
    kg_model.vocab_hash = (int *)calloc(kg_model.vocab_hash_size, sizeof(int));
    if (kg_model.read_vocab_file[0] != 0) ReadKgVocab();
    else {
        printf("please init kg model!");
        return;
    }
}

void InitJointModel(){
    joint_model.file_size = 0;
    joint_model.train_items = text_model.train_items;
    joint_model.item_count_actual = 0;
    joint_model.alpha = alpha;
    joint_model.vocab_max_size = kg_model.vocab_size;
    
    //init title using kg_model
    joint_model.vocab_hash_size = 30000000;  // Maximum items in the vocabulary 30m*0.7=21m
    joint_model.vocab_hash = (int *)calloc(joint_model.vocab_hash_size, sizeof(int));
    joint_model.vocab = (struct vocab_title *)calloc(joint_model.vocab_max_size, sizeof(struct vocab_title));
    if (joint_model.read_vocab_file[0]!=0) {
        ReadTitleVocab();
        initTitleVocab();
        InitTitleNet();
        if(joint_model.read_vector_file[0]!=0)
            LoadTitleVector();
    }
    else{
        printf("please input title file path!");
        return;
    }
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

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("Joint word&entity VECTOR training toolkit v 0.1c\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train_text <file>\n");
        printf("\t\tUse text data from <file> to train the text model and anchor model\n");
        printf("\t-train_kg <file>\n");
        printf("\t\tUse knowledge data from <file> to train the knowledge model\n");
        printf("\t-train_anchor <file>\n");
        printf("\t\tUse anchor data from <file> to train align model\n");
        printf("\t-output_path <path>\n");
        printf("\t\tUse <path> to save the resulting vectors \n");
        printf("\t-read_text_path <path>\n");
        printf("\t\tUse <path> to read the text vocab and word embeddings.\n");
        printf("\t-read_kg_path <path>\n");
        printf("\t\tUse <path> to read the entity vocab and entity embeddings.\n");
        printf("\t-read_title_path <path>\n");
        printf("\t\tUse <path> to read the entity title vocab.\n");
        printf("\t-save_text_path <path>\n");
        printf("\t\tUse <path> to save the text vocab and word embedding\n");
        printf("\t-save_kg_path <path>\n");
        printf("\t\tUse <path> to save the entity vocab and entity embedding\n");
        printf("\t-save_title_path <path>\n");
        printf("\t\tUse <path> to save the entity title vocab.\n");
        printf("\t-cluster_threshold <float>\n");
        printf("\t\tSet the vector's cosine similairty with a cluster larger than <float>, it will be assigned to the cluster; default is -0.5\n");
        printf("\t-max_sense_num <int>\n");
        printf("\t\tSet max sense number of the label for NIL; default is 10\n");
        printf("\t-text_sense <int>\n");
        printf("\t\tuse text model to update title sense if 1, otherwise 0. default is 1.\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word  / entity vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between (anchor) words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of (anchor) words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
        return 0;
    }
    
    text_model.read_vocab_file[0] = 0;
    text_model.read_vector_file[0] = 0;
    kg_model.read_vocab_file[0] = 0;
    joint_model.read_vocab_file[0] = 0;
    joint_model.read_vector_file[0] = 0;
    output_path[0]=0;
    mapping_title.map_file[0] = 0;
    if ((i = ArgPos((char *)"-train_text", argc, argv)) > 0) strcpy(text_model.train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-output_path", argc, argv)) > 0) strcpy(output_path, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_word_vocab", argc, argv)) > 0) strcpy(text_model.read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_word_vector", argc, argv)) > 0) strcpy(text_model.read_vector_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_kg_vocab", argc, argv)) > 0) strcpy(kg_model.read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_title_vocab", argc, argv)) > 0) strcpy(joint_model.read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_title_vector", argc, argv)) > 0) strcpy(joint_model.read_vector_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read_map_file", argc, argv)) > 0) strcpy(mapping_title.map_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-cluster_threshold", argc, argv)) > 0) cluster_threshold = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-max_sense_num", argc, argv)) > 0) max_sense_num = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-save_iter", argc, argv)) > 0) save_iter = atoi(argv[i + 1]);
    
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    
    if(output_path[0]!=0){
        sprintf(text_model.output_file, "%svectors_word_NIL", output_path);
        sprintf(joint_model.output_file, "%svectors_title_NIL", output_path);
    }
    
    InitTextModel();
    InitKgModel();
    InitJointModel();
    if(mapping_title.map_file[0]!=0)
        ReadMappingTitle();
    
    //start self training
    local_iter = 0;
    if (save_iter <=0 || save_iter > iter) save_iter = iter;
    while(local_iter<iter){
        local_iter++;
        printf("Start jointly training the %d time... ", local_iter);
        TrainTextModel();
        if (local_iter%save_iter==0){
            printf("saving results...\n");
            SaveWordVector(local_iter);
            SaveTitleVector(local_iter);
        }
    }
    return 0;
}
