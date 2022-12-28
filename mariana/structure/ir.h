/*
 *        (C) COPYRIGHT LeiNao Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : structure/ir.h
 * Authors    : wangliquan@zkln
 * Create Time: 2022-12-28:08:55:17
 * Description:
 *
 */

#ifndef __STRUCTURE_IR_H__
#define __STRUCTURE_IR_H__

#include <string>
#include <vector>
#include <iostream>
#include <functional>

#include <core/utils/arrary_ref.h>
#include <core/impl/shape.h>

namespace mariana {

struct Function;
using FuncPtr = std::shared_ptr<Function>;

struct OpKind {
    OpKind()=default;
    explicit OpKind(const std::string& op): op(op) {}
    bool operator==(const OpKind& rhs) const {
        return op == rhs.op;
    }
    bool operator!=(const OpKind& rhs) const {
        return !operator==(rhs);
    }
    static OpKind get(const std::string& name) {
        return OpKind(name);
    }
    std::string op;
};

inline std::ostream& operator<<(std::ostream& stream, const OpKind& op) {
    stream << op.op;
    return stream;
}

using OpList = ArrayRef<FuncPtr>;

class Node {
public:
    static bool enable_dynamic_shape() {
        return true;
    }
    Node(OpKind op, size_t num_outputs) : op_(op), num_outputs_(num_outputs) {}
    Node(OpKind op, OpList operands, std::vector<Shape>&& shapes,
         size_t num_outputs = 1) : Node(op, num_outputs) {
        shapes_.insert(shapes.end(),
                       std::make_move_iterator(shapes.begin()),
                       std::make_move_iterator(shapes.end()));
        for (auto& operand : operands) {
            if (!operand) {
                continue;
            }
            add_operand(operand);
        }
    }
    Node(OpKind op, OpList operands, const std::function<Shape()>& shape_fn,
         size_t num_outputs = 1) : Node(op, operands, std::vector<Shape>{}, num_outputs) {
        addComputedShape(shape_fn);
    }
    Node(OpKind op, OpList operands, size_t num_outputs = 1)
        : Node(op, operands, std::vector<Shape>{}, num_outputs) {}
    Node(OpKind op, Shape shape, size_t num_outputs = 1) : Node(op, num_outputs) {
        shapes_.push_back(std::move(shape));
    }
    ~Node()=default;
    void add_operand(FuncPtr func) {
        operands_.push_back(func);
    }
    const OpKind& op() const {
        return op_;
    }
    size_t num_outputs() const {
        return num_outputs_;
    }
    ArrayRef<Shape> shapes() const {
        return shapes_;
    }
    const Shape& shape(size_t output_index = 0) const {
        return shapes_.at(output_index);
    }
    void addComputedShape(const std::function<Shape()>& shape_fn) {
        shapes_.push_back(compute_shape(shape_fn));
    }
    Shape compute_shape(const std::function<Shape()>& shape_fn) {
        return shape_fn();
    }
    const std::vector<FuncPtr>& operands() const {
        return operands_;
    }
    FuncPtr operand(size_t i) const {
        return operands_.at(i);
    }
private:
    OpKind op_;
    size_t num_outputs_;
    std::vector<Shape> shapes_;
    std::vector<FuncPtr> operands_;
};

} // namespace mariana

#endif /* __STRUCTURE_IR_H__ */

