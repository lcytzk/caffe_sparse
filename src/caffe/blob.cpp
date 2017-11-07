#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    }
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    // TODO
    val_.reset(new SyncedMemory());
    one_val_.reset(new SyncedMemory());
    row_ptr_.reset(new SyncedMemory());
    col_ind_.reset(new SyncedMemory());
  }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : sparse_(false), capacity_(0) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : sparse_(false), capacity_(0) {
  Reshape(shape);
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_gpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_val_r() const {
  CHECK(val_r_);
  return (const Dtype*)val_r_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_val_r() const {
  CHECK(val_r_);
  return (const Dtype*)val_r_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_val() const {
  CHECK(val_);
  return (const Dtype*)val_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_val() const {
  CHECK(val_);
  return (const Dtype*)val_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_one_val() const {
  CHECK(one_val_);
  return (const Dtype*)one_val_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_one_val() const {
  CHECK(one_val_);
  return (const Dtype*)one_val_->gpu_data();
}

template <typename Dtype>
const int* Blob<Dtype>::cpu_row_ind() const {
  CHECK(row_ind_);
  return (const int*)row_ind_->cpu_data();
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_row_ind() const {
  CHECK(row_ind_);
  return (const int*)row_ind_->gpu_data();
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_row_ptr_r() const {
  CHECK(row_ptr_r_);
  return (const int*)row_ptr_r_->gpu_data();
}

template <typename Dtype>
const int* Blob<Dtype>::cpu_row_ptr_r() const {
  CHECK(row_ptr_r_);
  return (const int*)row_ptr_r_->cpu_data();
}

template <typename Dtype>
const int* Blob<Dtype>::cpu_row_ptr() const {
  CHECK(row_ptr_);
  return (const int*)row_ptr_->cpu_data();
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_row_ptr() const {
  CHECK(row_ptr_);
  return (const int*)row_ptr_->gpu_data();
}

template <typename Dtype>
const int* Blob<Dtype>::cpu_col_ind_r() const {
  CHECK(col_ind_r_);
  return (const int*)col_ind_r_->cpu_data();
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_col_ind_r() const {
  CHECK(col_ind_r_);
  return (const int*)col_ind_r_->gpu_data();
}

template <typename Dtype>
const int* Blob<Dtype>::cpu_col_ind() const {
  CHECK(col_ind_);
  return (const int*)col_ind_->cpu_data();
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_col_ind() const {
  CHECK(col_ind_);
  return (const int*)col_ind_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

// @liangchenye
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_val() {
  CHECK(val_);
  return static_cast<Dtype*>(val_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_val() {
  CHECK(val_);
  return static_cast<Dtype*>(val_->mutable_cpu_data());
}

template <typename Dtype>
int* Blob<Dtype>::mutable_gpu_row_ptr_r() {
  CHECK(row_ptr_r_);
  return static_cast<int*>(row_ptr_r_->mutable_gpu_data());
}

template <typename Dtype>
int* Blob<Dtype>::mutable_cpu_row_ptr_r() {
  CHECK(row_ptr_r_);
  return static_cast<int*>(row_ptr_r_->mutable_cpu_data());
}

template <typename Dtype>
int* Blob<Dtype>::mutable_gpu_row_ptr() {
  CHECK(row_ptr_);
  return static_cast<int*>(row_ptr_->mutable_gpu_data());
}

template <typename Dtype>
int* Blob<Dtype>::mutable_cpu_row_ptr() {
  CHECK(row_ptr_);
  return static_cast<int*>(row_ptr_->mutable_cpu_data());
}

template <typename Dtype>
int* Blob<Dtype>::mutable_gpu_col_ind_r() {
  CHECK(col_ind_r_);
  return static_cast<int*>(col_ind_r_->mutable_gpu_data());
}

template <typename Dtype>
int* Blob<Dtype>::mutable_cpu_col_ind_r() {
  CHECK(col_ind_r_);
  return static_cast<int*>(col_ind_r_->mutable_cpu_data());
}

template <typename Dtype>
int* Blob<Dtype>::mutable_gpu_col_ind() {
  CHECK(col_ind_);
  return static_cast<int*>(col_ind_->mutable_gpu_data());
}

template <typename Dtype>
int* Blob<Dtype>::mutable_cpu_col_ind() {
  CHECK(col_ind_);
  return static_cast<int*>(col_ind_->mutable_cpu_data());
}

template <typename Dtype>
int* Blob<Dtype>::mutable_gpu_one_val() {
  CHECK(one_val_);
  return static_cast<int*>(one_val_->mutable_gpu_data());
}

template <typename Dtype>
int* Blob<Dtype>::mutable_cpu_one_val() {
  CHECK(one_val_);
  return static_cast<int*>(one_val_->mutable_cpu_data());
}

// end

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    // @liangchenye
    if(Caffe::step() == 2 && sparse_) {
      //const Dtype* vv = (const Dtype*) val_->cpu_data();
      //const Dtype* dd = (const Dtype*) diff_->cpu_data();
      //const int* cc = (const int*) col_ind_->cpu_data();
      //const int* rr = (const int*) row_ind_->cpu_data();
      ////LOG(INFO) << "backward sparse start";
      //LOG(INFO) << cc[0];
      //LOG(INFO) << rr[0] * shape_[0] + cc[0];
      ////for(int i = 0; i < 10; ++i) {
        //printf("%f\t%f\t%f\n", vv[0], dd[rr[0] * shape_[0] + cc[0]], dd[cc[0] * shape_[1] + rr[0]]);
      ////}
      ////LOG(INFO) << "backward sparse add";
      //  //val_->mutable_gpu_data(),
      //  //row_ptr_->gpu_data();
      //  //col_ind_->gpu_data();
      //  //diff_->gpu_data();
      ////LOG(INFO) << "backward sparse add";
      ////LOG(INFO) << shape_string();
      caffe_gpu_sparse_add(
        shape_[1], shape_[0], nnz_,
        static_cast<Dtype*>(val_->mutable_gpu_data()),
        static_cast<const int*>(row_ind_->gpu_data()),
        static_cast<const int*>(col_ind_->gpu_data()),
        static_cast<const Dtype*>(diff_->gpu_data()));
      ////LOG(INFO) << "backward sparse add done";
      //vv = (const Dtype*) val_->cpu_data();
      ////for(int i = 0; i < 10; ++i) {
      ////  printf("%d:%f\t", i, vv[i]);
      ////}
        //printf("%f\n", vv[0]);
    } else {
      caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = gpu_diff();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = mutable_gpu_diff();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // read sparse @liangchenye
  if(proto.val_size() > 0) {
    nnz_ = proto.nnz();
    sparse_ = proto.sparse();
    // init sync mem
    val_.reset(new SyncedMemory(nnz_ * sizeof(Dtype)));
    one_val_.reset(new SyncedMemory(nnz_ * sizeof(int)));
    row_ptr_.reset(new SyncedMemory((shape_[1] + 1) * sizeof(int)));
    col_ind_.reset(new SyncedMemory(nnz_ * sizeof(int)));
    // end init
    LOG(INFO) << "nnz: " << nnz_ << "  sparse: " << sparse_;
    Dtype* val = mutable_cpu_val();
    int* one_val = mutable_cpu_one_val();
    int* col_ind = mutable_cpu_col_ind();
    int* row_ptr = mutable_cpu_row_ptr();
    for(int i = 0; i < nnz_; ++i) {
      val[i] = proto.val(i);
      one_val[i] = 1;
      col_ind[i] = proto.col_ind(i);
    }
    for(int i = 0; i < shape_[1] + 1; ++i) {
      row_ptr[i] = proto.row_ptr(i);
    }
    if(Caffe::step() == 2) {
        row_ind_.reset(new SyncedMemory(nnz_ * sizeof(int)));
        caffe_gpu_csr2coo(gpu_row_ptr(), nnz_, shape_[1], (int*)row_ind_->mutable_gpu_data());
    }
  } else {
    // copy data
    Dtype* data_vec = mutable_cpu_data();
    if (proto.double_data_size() > 0) {
      CHECK_EQ(count_, proto.double_data_size());
      for (int i = 0; i < count_; ++i) {
        data_vec[i] = proto.double_data(i);
      }
    } else {
      CHECK_EQ(count_, proto.data_size());
      for (int i = 0; i < count_; ++i) {
        data_vec[i] = proto.data(i);
      }
    }
  }
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  if(sparse_) {
    // TODO  step 1 need to prune the weight
    proto->set_sparse(true);
    LOG(INFO) << "set sparse as true: " << proto->sparse();
    double* val = NULL;
    int* rowPtr = NULL;
    int* colInd = NULL;
    int nnz;
    if(Caffe::step() == 1) {
      caffe_dense2sparse(shape_[1], shape_[0], gpu_data(), 95, &nnz, &val, &colInd, &rowPtr);
    } else {
      val = mutable_cpu_val();
      rowPtr = mutable_cpu_row_ptr();
      colInd = mutable_cpu_col_ind();
      nnz = nnz_;
    }
    LOG(INFO) << "nnz: " << nnz;
    proto->set_nnz(nnz);
    for(int i = 0; i < nnz; ++i) {
      proto->add_val(val[i]);
      proto->add_col_ind(colInd[i]);
    }
    for(int i = 0; i < shape_[1] + 1; ++i) {
      proto->add_row_ptr(rowPtr[i]);
    }
  } else {
    const double* data_vec = cpu_data();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_data(data_vec[i]);
    }
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  proto->clear_val();
  proto->clear_row_ptr();
  proto->clear_col_ind();
  if(sparse_) {
    // TODO  step 1 need to prune the weight
    proto->set_sparse(true);
    LOG(INFO) << "set sparse as true: " << proto->sparse();
    float* val = NULL;
    int* rowPtr = NULL;
    int* colInd = NULL;
    int nnz;
    if(Caffe::step() == 1) {
      caffe_dense2sparse(shape_[1], shape_[0], gpu_data(), 90, &nnz, &val, &colInd, &rowPtr);
    } else {
        val = mutable_cpu_val();
        rowPtr = mutable_cpu_row_ptr();
        colInd = mutable_cpu_col_ind();
        nnz = nnz_;
    }
    //CHECK(val);
    //CHECK(rowPtr);
    //CHECK(colInd);
    //LOG(INFO) << "sparse: " << sparse_;
    //LOG(INFO) << "dense 2 sparse done, ready to save.";
    //LOG(INFO) << "count: " << count_ << " shape[0]: " << shape_[0] << " shape[1]: " << shape_[1] << " shape[2]: " << shape_[2] << " shape[3]: " << shape_[3] ;
    //LOG(INFO) << "nnz: " << nnz << " val: " << val[0] << " col: " << colInd[0];
    LOG(INFO) << "count: " << count_ << "  nnz: " << nnz;
    proto->set_nnz(nnz);
    for(int i = 0; i < nnz; ++i) {
      proto->add_val(val[i]);
      proto->add_col_ind(colInd[i]);
    }
    //LOG(INFO) << "ready to save shape.";
    // shape[0] need to add one because of the csr format, at the end of row ptr is a end number
    for(int i = 0; i < shape_[1] + 1; ++i) {
      proto->add_row_ptr(rowPtr[i]);
    }
  } else {
    const float* data_vec = cpu_data();
    for (int i = 0; i < count_; ++i) {
      proto->add_data(data_vec[i]);
    }
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe

