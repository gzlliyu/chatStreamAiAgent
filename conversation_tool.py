from datetime import datetime

from langchain.tools import BaseTool


class OrderSearchTool(BaseTool):
    name = "订单查询"
    description = "查询订单信息，入参是订单编号（格式是DJ加数字），返回值是订单信息,如果用户没有提供订单信息先不要使用本工具,请以客服的角色和用户沟通他要咨询哪个订单，提供下订单号"
    use_id: int

    def _run(
            self,
            order_code: str,
    ) -> str:
        print(f'[OrderSearch][{datetime.now()}]用户订单查询，doctor_id={self.use_id},order_code={order_code}')
        # TODO LZK 调用后台接口查询订单信息
        if order_code is not None:
            tmp_order = order_code.strip().lower()
            if not tmp_order.startswith('dj'):
                return '请以客服的角色和用户沟通他要咨询哪个订单，提供下订单号'
        return '患者姓名：张三；年龄：20，性别：男;订单状态：已发货'


class ExpressChangeTool(BaseTool):
    name = "物流修改"
    description = "物流修改，入参是订单编号order_code（格式是DJ加数字）和新的物流地址new_address用半角逗号组成的字符串（新物流地址未提供请用xxx替代地址）" \
                  "，返回值是物流修改的客服工单号，使用本工具前请确保获取到了订单编号和新的物流地址，如果有缺失请先别使用本工具，先告诉用户提供必要信息。"
    user_id: int

    def _run(
            self,
            order_code_and_new_address: str,
    ) -> str:
        print(
            f'[ExpressChange][{datetime.now()}]物流修改:doctor_id={self.user_id},'
            f'order_code_and_new_address={order_code_and_new_address}')
        try:
            order_code = order_code_and_new_address.split(',')[0]
            new_address = order_code_and_new_address.split(',')[1]
            if new_address == 'xxx':
                return '请先提供新的订单地址！'
        except IndexError:
            return '参数缺失，请确认订单号和物流地址是否提供！'
        except:
            return '物流修改失败，请联系人工客服！'
        # TODO LZK 调用业务后台接口发起物流修改代办
        return '您的物流修改流程已发起，工单号：gd-001，我们将尽快处理您的请求。'


class OrderCancelTool(BaseTool):
    name = "订单取消工具"

    description = """帮助用户取消订单，入参是订单编号order_code（格式是DJ加数字），返回值是订单取消的客服工单号，
    使用本工具前请确保获取到了订单编号，如果有缺失请先别使用本工具先告诉用户提供必要信息。"""

    user_id: int

    def _run(
            self,
            order_code: str,
    ) -> str:
        print(
            f'[OrderCancel][{datetime.now()}]订单取消:doctor_id={self.user_id},order_code={order_code}')
        # TODO LZK 调用业务后台接口发起订单取消的代办
        return '您的订单取消流程已发起，工单号：gd-002，我们将尽快处理您的请求。'


class WeatherSearchTool(BaseTool):
    name = "天气查询"
    description = "查询天气信息，入参是地理位置信息，返回值是具体天气情况，如果用户未提供要查询天气的位置，请先别使用本工具先让用户提供具体位置。"
    use_id: int

    def _run(
            self,
            location: str
    ) -> str:
        return f'{location}今天小雨，东北风3级，天气潮湿，不适宜洗车，出门记得带伞～'
