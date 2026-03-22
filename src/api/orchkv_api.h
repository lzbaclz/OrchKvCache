#ifndef ORCHKV_API_H
#define ORCHKV_API_H

#include "../core/kv_types.h"
#include "../core/kv_request.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Full API – stub declarations, implementation in A8 */

int orchkv_init(const orchkv_config_t *config);
int orchkv_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* ORCHKV_API_H */
