#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <vector>

// #define HBM_COLUMN_VERBOSE

using namespace std;


#define ALIGNMENT BYTES_IN_LINE * 2
const uint32_t HBM_PORT_LINES = 2048; // 2048 lines

#define NUM_KERNEL 32

typedef struct {
  uint32_t m_id;
  int m_value;
} tuple_t;

template <typename T> class hbm_column {
private:
  const uint32_t ITEMS_IN_LINE = BYTES_IN_LINE / sizeof(T);

  // CPU side
  T *m_base;
  uint32_t m_capacity_items;
  uint32_t m_num_items;
  bool m_stride_wise;
  uint32_t len_stride;
  bool m_for_join;

public:
  // HBM side
  uint32_t m_num_partitions;
  uint64_t m_base_hbm_offset;
  uint32_t m_total_num_lines;
  vector<uint64_t> m_hbm_offset;
  vector<uint32_t> m_num_lines;
  vector<uint32_t> m_offset;

  hbm_column(uint32_t capacity_items, uint64_t hbm_offset) {
    m_base_hbm_offset = hbm_offset;
    m_num_items = 0;
    m_base = NULL;
    m_stride_wise = false;
    m_for_join = false;
    column_realloc(capacity_items);
    set_partitions(NUM_KERNEL);
  }

  hbm_column(uint32_t capacity_items, uint64_t hbm_offset, bool stride_wise,
             uint64_t len_stride) {
    m_base_hbm_offset = hbm_offset;
    m_num_items = 0;
    m_base = NULL;
    m_stride_wise = stride_wise;
    len_stride = len_stride;
    m_for_join = false;
    column_realloc(capacity_items);
    set_partitions(NUM_KERNEL);
  }

  hbm_column(uint32_t capacity_items, uint64_t hbm_offset, bool stride_wise,
             uint64_t len_stride, bool for_join) {
    m_base_hbm_offset = hbm_offset;
    m_num_items = 0;
    m_base = NULL;
    m_stride_wise = stride_wise;
    len_stride = len_stride;
    m_for_join = for_join;
    column_realloc(capacity_items);
    set_partitions(NUM_KERNEL);
  }

  ~hbm_column() { free(m_base); }

  bool get_stride_wise() { return m_stride_wise; }

  void set_partitions(uint32_t num_partitions) {
    m_num_partitions = num_partitions;
    m_hbm_offset.resize(m_num_partitions);
    m_num_lines.resize(m_num_partitions);
    m_offset.resize(m_num_partitions);
    uint32_t assigned_num_lines = 0;
    for (uint32_t i = 0; i < m_num_partitions; i++) {
      if (m_stride_wise) {
        if (m_for_join) {
          m_hbm_offset[i] = m_base_hbm_offset + 2 * i * len_stride;
        } else {
          m_hbm_offset[i] = m_base_hbm_offset + i * len_stride;
        }
      } else {
        m_hbm_offset[i] = m_base_hbm_offset + assigned_num_lines;
      }

      // m_offset[i] = assigned_num_lines;
      if (i == m_num_partitions - 1) {
        m_num_lines[i] = m_total_num_lines - assigned_num_lines;
      } else {
        m_num_lines[i] = m_total_num_lines / m_num_partitions;
      }
      // for 256-bit
      // m_num_lines[i] += (m_num_lines[i]%2 == 0) ? 0 : 1;
      // #ifdef HBM_COLUMN_VERBOSE
      //             if (m_num_partitions > 1) {
      //                 cout << "hbm_column, set_partitions: 0x" << hex <<
      //                 m_hbm_offset[i] << dec << ", m_num_lines[" << i << "]:
      //                 " << m_num_lines[i] << endl;
      //             }
      // #endif
      assigned_num_lines += m_num_lines[i];
    }
  }

  void column_realloc(uint32_t new_capacity) {
    T *new_base;
    m_capacity_items = new_capacity;
    m_total_num_lines =
        new_capacity / ITEMS_IN_LINE + (new_capacity % ITEMS_IN_LINE > 0);
    // for 256-bit
    // m_total_num_lines += (m_total_num_lines%2 == 0) ? 0 : 1;

    // FIXME: do alignment for BAT? time cost?
    posix_memalign((void **)&new_base, ALIGNMENT,
                   m_total_num_lines * BYTES_IN_LINE);
    if (m_capacity_items < m_num_items) {
      m_num_items = m_capacity_items;
    }
    for (uint32_t i = 0; i < m_num_items; i++) {
      new_base[i] = m_base[i];
    }
    free(m_base);
    m_base = new_base;
  }

  uint32_t get_num_items() { return m_num_items; }

  T *get_base() { return m_base; }

  T get_item(uint32_t index) {
    if (index < m_total_num_lines * ITEMS_IN_LINE) {
      return m_base[index];
    } else {
      return (T)0;
    }
  }

  void set_item(T value, uint32_t index) {
    if (index < m_capacity_items) {
      m_base[index] = value;
    }
  }

  void sort_items() {
    sort((uint32_t *)m_base, ((uint32_t *)m_base) + m_num_items);
  }

  void populate_int_column(uint32_t num_items, char unique, char shuffle,
                           int dummy) {
    if (num_items > m_capacity_items) {
      column_realloc(num_items);
    }

    for (uint32_t i = 0; i < num_items; i++) {
      m_base[i] = (unique == 'u') ? i : rand() % num_items;
    }
    for (uint32_t i = num_items; i < m_total_num_lines * ITEMS_IN_LINE; i++) {
      m_base[i] = dummy;
    }
    if (shuffle == 's') {
      for (uint32_t i = 0; i < num_items; i++) {
        uint32_t index = rand() % num_items;
        uint32_t temp = m_base[i];
        m_base[i] = m_base[index];
        m_base[index] = temp;
      }
    }
    m_num_items = num_items;
  }

  void append(T value) {
    if (m_num_items >= m_capacity_items) {
      column_realloc(2 * m_capacity_items);
    }
    m_base[m_num_items++] = value;
  }
};