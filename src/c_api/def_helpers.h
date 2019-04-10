#define PRINT_ARRAY(VAR, DIM, END_VAR, FORMAT) \
    BUFFER_DEF(VAR, DIM) \
    FOREACH(VAR, END_VAR, FORMAT)

#define BUFFER_DEF(VAR, DIM) \
    char VAR##_string_[DIM] = "{"; \
    char* VAR##_string = &VAR##_string_[1];


#define FOREACH(VAR, END_VAR, FORMAT) \
    FOREACH_NAMED(VAR, VAR, END_VAR, FORMAT)

#define FOREACH_NAMED(VAR_NAME, VAR, END_VAR, FORMAT) \
  for(unsigned int i = 0; i < (END_VAR);i++) \
  { VAR_NAME##_string += sprintf(VAR_NAME##_string, FORMAT, (VAR)[i]); } \
  if (END_VAR) \
  { VAR_NAME##_string[-2] = '}'; VAR_NAME##_string[-1] = ','; } \
  VAR_NAME##_string[0] = END_VAR ? ' ' : '}';  VAR_NAME##_string[1] = '\0';

#define ARG(ARG_) (#ARG_ "=") << ARG_ << ", "
#define ARR(ARR_) (#ARR_ "=") << ARR_##_string_

